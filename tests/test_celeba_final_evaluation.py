"""Final assessment must use accepted checkpoints and only final sample roles."""
import copy
import json

import numpy as np
import pytest
import torch

from experiments.CelebA import final_evaluation as final
from experiments.CelebA import model_selection as selection
from experiments.CelebA.selection_training import fit_model


def test_selected_tasks_require_all_pairs_repeats_and_fixed_settings():
    cfg = {'levels': ['feature', 'pixel'], 'branches': ['lower', 'upper'],
           'losses': ['hellinger', 'js'], 'output_alphas': [.5, 2, 4], 'repeats': 5}
    protocol = {'config': cfg, 'tasks': selection.grid(cfg)}
    chosen = final.selected_tasks(protocol)
    assert len(chosen) == 20
    assert all(t['loss'] == 'js' for t in chosen)
    for broken in (chosen[:-1], chosen + [chosen[0]]):
        with pytest.raises(ValueError, match='four paired comparisons'):
            final.selected_tasks({**protocol, 'tasks': broken})


@pytest.mark.parametrize('level', ['feature', 'pixel'])
def test_real_checkpoint_final_only_and_tamper_rejection(tmp_path, monkeypatch, level):
    torch.set_num_threads(1)
    cfg = {'seed': 7, 'bins': 20, 'ci_alpha': .05, 'repeats': 1,
           'feature': {'input_dimension': 3, 'hidden_dimension': 4, 'batch_size': 4,
                       'max_epochs': 1, 'min_epochs': 1, 'patience': 1},
           'pixel': {'ndf': 2, 'batch_size': 4, 'max_epochs': 1, 'min_epochs': 1,
                     'patience': 1, 'bn_freeze_epoch': 1}}
    task = dict(representation=level, branch='lower', architecture='baseline',
                candidate='baseline_js', loss='js', output_alpha=final.SETTINGS[level][1], repeat=0)
    parent, output = tmp_path / 'parent', tmp_path / 'final'
    output.mkdir()
    checkpoint = selection.task_directory(parent, task)
    checkpoint.mkdir(parents=True)
    rng = np.random.default_rng(7)
    def values():
        return (rng.normal(size=(8, 3)).astype('float32') if level == 'feature'
                else rng.integers(0, 256, (8, 3, 64, 64), dtype='uint8'))
    development = {f'{role}_{side}': values() for role in selection.TRAIN_ROLES for side in ('p', 'q')}
    fit_model(task, cfg, development, checkpoint, 'cpu')
    parent_hash = 'frozen-parent'
    selection.write(checkpoint / 'COMPLETE.json', dict(task_identity=selection.task_identity(task),
        protocol_sha256=parent_hash, files={n: selection.sha(checkpoint / n) for n in selection.FIT_FILES}))
    selection.write(output / 'protocol.json', {})
    selection.write(output / 'freeze.json', {})
    selection.write(output / 'data/evaluation_metadata.json', {})
    selection.seal(output / 'evaluation_data.json', dict(
        metadata_sha256=selection.sha(output / 'data/evaluation_metadata.json'),
        freeze_sha256=selection.sha(output / 'freeze.json')))
    protocol = dict(parent=str(parent), parent_protocol_sha256=parent_hash, config=cfg, tasks=[task],
                    checkpoint_receipts={str(checkpoint): selection.sha(checkpoint / 'COMPLETE.json')},
                    scope_caveat='Retrospective smoke.', selection_limitation='Smoke only.')
    monkeypatch.setattr(final, 'locked', lambda out: copy.deepcopy(protocol))
    calls = []
    arrays = {role: {side: values() for side in ('p', 'q')} for role in selection.TEST_ROLES}
    def load(out, representation, branch, roles):
        calls.append(tuple(roles))
        assert roles == selection.TEST_ROLES
        return arrays
    monkeypatch.setattr(final, 'load_arrays', load)
    with pytest.raises(ValueError, match='incomplete'):
        final.report(output, require_complete=True)
    assert not (output / 'COMPLETE.json').exists()
    final.evaluate(output, 0, 'cpu')
    assert calls == [selection.TEST_ROLES]
    final.evaluate(output, 0, 'cpu')  # completion receipts make successful evaluations idempotent
    assert len(calls) == 1
    final.report(output, require_complete=True)
    assert json.loads((output / 'COMPLETE.json').read_text())['evaluations'] == 1
    directory = selection.task_directory(output, task, evaluation=True)
    (directory / 'metrics.json').write_text('{}')
    with pytest.raises(ValueError, match='artifact changed'):
        final.report(output, require_complete=True)
