#!/usr/bin/env python3
"""Refresh presentation from verified frozen fits without changing training source."""
import argparse
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.CelebA.model_selection import collect, read_sealed, sha, write
from experiments.CelebA.selection_only_report import build_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--accepted-run', type=Path, required=True)
    args = parser.parse_args()
    output, accepted = args.output.resolve(), args.accepted_run.resolve()
    protocol = read_sealed(output / 'protocol.json')
    for name, digest in protocol['files'].items():
        if sha(output / name) != digest:
            raise ValueError(f'Frozen selection source/data changed: {name}')
    final_protocol = read_sealed(accepted / 'protocol.json')
    freeze = read_sealed(accepted / 'freeze.json')
    if (final_protocol['parent'] != str(output)
            or final_protocol['parent_protocol_sha256'] != sha(output / 'protocol.json')
            or freeze['protocol_sha256'] != sha(accepted / 'protocol.json')):
        raise ValueError('Accepted final settings refer to another selection run')
    accepted_selection = {'selections': {}}
    for task in final_protocol['tasks']:
        key = f"{task['representation']}/{task['branch']}"
        choice = {k: task[k] for k in ('loss', 'output_alpha', 'candidate')}
        if accepted_selection['selections'].setdefault(key, {'chosen': choice}) != {'chosen': choice}:
            raise ValueError('Accepted tasks have inconsistent settings')
    rows = collect(output, protocol, protocol['tasks'], require_complete=False)
    selected = read_sealed(output / 'baseline_selection.json') if (output / 'baseline_selection.json').exists() else None
    build_report(output, protocol, rows, selected, accepted_selection=accepted_selection)
    sources = {}
    for name in ('refresh_selection_presentation.py', 'selection_only_report.py', 'selection_report.py'):
        source = Path(__file__).with_name(name)
        target = output / 'presentation_source' / name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(source, target)
        sources[name] = sha(target)
    write(output / 'presentation_receipt.json', dict(
        protocol_sha256=sha(output / 'protocol.json'),
        accepted_freeze_sha256=sha(accepted / 'freeze.json'), verified_fits=len(rows),
        renderer_sources=sources,
        artifacts={name: sha(output / name) for name in (
            'RESULTS.md', 'report_status.json', 'per_fit.csv', 'per_candidate.csv',
            'selection_grid_feature.png', 'selection_grid_feature.pdf',
            'selection_grid_pixel.png', 'selection_grid_pixel.pdf')}))
    print(f'Refreshed presentation from {len(rows)} verified fits; frozen training source unchanged.')


if __name__ == '__main__':
    main()
