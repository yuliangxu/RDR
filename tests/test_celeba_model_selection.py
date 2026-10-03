"""Boundaries protecting the adaptive selection and final-assessment protocol."""
import copy
from pathlib import Path

import pytest

from experiments.CelebA import model_selection as selection


def config():
    return selection.read(Path(selection.__file__).with_name('selection_config.json'))


def rows_for(cfg, architecture='baseline', settings=None):
    return [{**t, 'brier': .10 + .001 * t['repeat'], 'local_gap': .03,
             'supported_mass': 1.0} for t in selection.grid(cfg, architecture, settings=settings)]


def test_baseline_grid_is_paired_both_representations_and_pairs():
    cfg = config()
    tasks = selection.grid(cfg)
    assert len(tasks) == 320
    assert len({selection.task_identity(t) for t in tasks}) == 320
    for level in cfg['levels']:
        for branch in cfg['branches']:
            assert len([t for t in tasks if t['representation'] == level and t['branch'] == branch]) == 80


def test_sensitivity_uses_same_union_on_both_pairs_and_limits_leaders():
    cfg = config()
    rows = rows_for(cfg)
    for row in rows:
        if row['branch'] == 'lower' and row['loss'] == 'js' and row['output_alpha'] == 2:
            row['brier'] -= .01
        if row['branch'] == 'upper' and row['loss'] == 'kl' and row['output_alpha'] == .5:
            row['brier'] -= .01
    choices = selection.selections(rows, cfg)
    settings = selection.sensitivity_settings(choices, cfg)
    assert all(2 <= len(v) <= 4 for v in settings.values())
    tasks = selection.grid(cfg, 'small', settings=settings)
    for level in cfg['levels']:
        groups = [{t['candidate'] for t in tasks if t['representation'] == level and t['branch'] == b}
                  for b in cfg['branches']]
        assert groups[0] == groups[1]


def test_architecture_expansion_requires_size_consistency_support_and_pairing():
    cfg = config()
    base = rows_for(cfg)
    selected = selection.selections(base, cfg)
    small = rows_for(cfg, 'small')
    assert not selection.expansion_decision(base, small, selected, cfg)['expanded_representations']
    improved = copy.deepcopy(small)
    for row in improved:
        if row['representation'] == 'pixel' and row['branch'] == 'upper':
            row['brier'] -= .003
    assert selection.expansion_decision(base, improved, selected, cfg)['expanded_representations'] == ['pixel']
    unsupported = [{**r, 'supported_mass': .98} for r in improved]
    assert not selection.expansion_decision(base, unsupported, selected, cfg)['expanded_representations']
    with pytest.raises(ValueError, match='complete paired repeats'):
        selection.expansion_decision(base, improved[:-1], selected, cfg)


def test_sealed_records_reject_tamper_and_changed_choice(tmp_path):
    path = tmp_path / 'freeze.json'
    selection.seal(path, {'choice': 'a'})
    selection.seal(path, {'choice': 'a'})
    with pytest.raises(ValueError, match='Refusing to change'):
        selection.seal(path, {'choice': 'b'})
    selection.write(path, {'choice': 'b'})
    with pytest.raises(ValueError, match='changed'):
        selection.read_sealed(path)


def test_missing_or_tampered_fit_cannot_enter_selection(tmp_path):
    cfg = config()
    task = selection.grid(cfg)[0]
    selection.write(tmp_path / 'protocol.json', {'config': cfg})
    with pytest.raises(ValueError, match='Missing completed task'):
        selection.collect(tmp_path, {'config': cfg}, [task])
    directory = selection.task_directory(tmp_path, task)
    directory.mkdir(parents=True)
    for name in selection.FIT_FILES:
        (directory / name).write_text('{}')
    selection.write(directory / 'metrics.json', task)
    selection.write(directory / 'COMPLETE.json', dict(task_identity=selection.task_identity(task),
                    protocol_sha256=selection.sha(tmp_path / 'protocol.json'),
                    files={name: selection.sha(directory / name) for name in selection.FIT_FILES}))
    assert len(selection.collect(tmp_path, {'config': cfg}, [task])) == 1
    (directory / 'model.pt').write_text('changed')
    with pytest.raises(ValueError, match='artifact changed'):
        selection.collect(tmp_path, {'config': cfg}, [task])


def test_shared_choice_requires_each_pair_eligible():
    cfg = config()
    choices = selection.selections(rows_for(cfg), cfg)
    for level in cfg['levels']:
        for row in choices[f'{level}/lower']['ranking']:
            row['eligible'] = row['loss'] == 'js'
        for row in choices[f'{level}/upper']['ranking']:
            row['eligible'] = row['loss'] == 'kl'
    assert all(v['candidate'] is None for v in selection.shared_configurations(choices, cfg).values())
