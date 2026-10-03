"""Guard the final artifact boundary and immutable report exports."""
import hashlib
import json
from pathlib import Path

import pytest

from experiments.CelebA.package_paper import build, json_bytes, verify


@pytest.fixture
def export(tmp_path):
    source = tmp_path / 'source'
    output = tmp_path / 'output'
    policy = tmp_path / 'policy.json'
    records = {}
    for index in range(32):
        for extension in ('png', 'pdf'):
            name = f'run/figure_{index:02d}.{extension}'
            data = f'unchanged example {index} {extension}'.encode()
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            records[name] = dict(source=f'/cwork/yx306/RDR/CelebA/{name}',
                                 kind='byte_identical_copy', bytes=len(data),
                                 sha256=hashlib.sha256(data).hexdigest())
    before = b'Value: 0.123; [plot](old.png)\n'
    after = before.replace(b'old.png', b'figure_00.png')
    (source / 'run/RESULTS.md').write_bytes(before)
    records['run/RESULTS.md'] = dict(source='/cwork/yx306/RDR/CelebA/run/RESULTS.md',
        kind='report_link_adjustment', source_sha256=hashlib.sha256(before).hexdigest(),
        sha256=hashlib.sha256(after).hexdigest(), bytes=len(after),
        replacements=[['old.png', 'figure_00.png']])
    policy.write_bytes(json_bytes(dict(files=records, figure_pairs=32)))
    build(source, output, policy)
    return source, output, policy


def test_final_export_and_report_correction(export):
    _, output, policy = export
    verify(output, policy)
    assert (output / 'run/RESULTS.md').read_text() == 'Value: 0.123; [plot](figure_00.png)\n'


def test_reject_changed_numeric_or_figure_bytes(export):
    _, output, policy = export
    (output / 'run/figure_00.png').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Changed paper artifact'):
        verify(output, policy)


def test_reject_superseded_unlisted_artifacts(export):
    _, output, policy = export
    (output / 'run/old_figure.png').write_bytes(b'obsolete')
    with pytest.raises(ValueError, match='Unlisted or missing'):
        verify(output, policy)


def test_rebuild_rejects_changed_upstream(export):
    source, output, policy = export
    (source / 'run/RESULTS.md').write_text('Value: 0.999; [plot](old.png)\n')
    with pytest.raises(ValueError, match='Changed upstream'):
        build(source, output, policy)
