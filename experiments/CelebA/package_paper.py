#!/usr/bin/env python3
"""Build and verify only the finalized CelebA paper artifacts.

The checked-in policy pins every upstream artifact. Report-only link/label
corrections are explicit; numeric tables, images and archived sources keep
identical bytes. Raw data and fitted models remain in the research archive.
"""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / 'results/CelebA'
DEFAULT_POLICY = Path(__file__).with_name('paper_files.json')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()


def put(path, data):
    if path.exists():
        require(path.read_bytes() == data, f'Refusing to overwrite changed file: {path}')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def source_archive(root, members):
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode='wb', filename='', mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode='w') as archive:
            for name, digest in sorted(members.items()):
                path = root / name
                require(not path.is_symlink() and sha(path) == digest,
                        f'Changed frozen source: {path}')
                info = tarfile.TarInfo(name)
                info.size = path.stat().st_size
                info.mode = 0o644
                with path.open('rb') as stream:
                    archive.addfile(info, stream)
    return buffer.getvalue()


def build(base, output, policy=DEFAULT_POLICY):
    manifest = read(policy)
    original_base = Path('/cwork/yx306/RDR/CelebA')
    for name, record in manifest['files'].items():
        source = base / Path(record['source']).relative_to(original_base)
        if record['kind'] == 'deterministic_source_archive':
            data = source_archive(source, record['members'])
        else:
            require(sha(source) == record.get('source_sha256', record['sha256']),
                    f'Changed upstream artifact: {source}')
            data = source.read_bytes()
            for before, after in record.get('replacements', []):
                require(before.encode() in data, f'Missing report replacement: {before}')
                data = data.replace(before.encode(), after.encode())
        require(hashlib.sha256(data).hexdigest() == record['sha256'] and
                len(data) == record['bytes'], f'Export differs from final policy: {name}')
        put(output / name, data)
    put(output / 'manifest.json', json_bytes(manifest))
    put(output / 'manifest.sha256', (sha(output / 'manifest.json') + '\n').encode())
    verify(output, policy)


def verify(output, policy=DEFAULT_POLICY):
    manifest = read(output / 'manifest.json')
    require(sha(output / 'manifest.json') == (output / 'manifest.sha256').read_text().strip(),
            'Changed package manifest')
    require(manifest == read(policy), 'Package differs from the final export policy')
    for name, record in manifest['files'].items():
        path = output / name
        require(path.stat().st_size == record['bytes'] and sha(path) == record['sha256'],
                f'Changed paper artifact: {name}')
        if record['kind'] == 'deterministic_source_archive':
            with tarfile.open(path, 'r:gz') as archive:
                require(set(archive.getnames()) == set(record['members']), 'Changed source members')
                for member, digest in record['members'].items():
                    require(hashlib.sha256(archive.extractfile(member).read()).hexdigest() == digest,
                            f'Changed frozen source: {member}')
    documented = set(manifest['files'])
    # Root-level guides/audits supplement the pinned scientific evidence.
    actual = {str(p.relative_to(output)) for p in output.glob('*/*') if p.is_file()}
    actual.update(str(p.relative_to(output)) for d in output.iterdir() if d.is_dir()
                  for p in d.rglob('*') if p.is_file())
    require(actual == documented, 'Unlisted or missing run artifacts: ' +
            str(sorted(actual.symmetric_difference(documented))))
    png = {str(Path(n).with_suffix('')) for n in documented if n.endswith('.png')}
    pdf = {str(Path(n).with_suffix('')) for n in documented if n.endswith('.pdf')}
    require(png == pdf and len(png) == manifest['figure_pairs'] == 32,
            'Expected exactly 32 matching PNG/PDF figure pairs')
    size = sum(r['bytes'] for r in manifest['files'].values()) / 1024**2
    print(f'Verified {len(documented)} artifacts ({size:.2f} MiB), 32 final PNG/PDF pairs; '
          'main 20/20, null 25/25, selection 319/320 (one recorded failure).')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('build', 'verify'))
    parser.add_argument('--source-root', type=Path, default=Path('/cwork/yx306/RDR/CelebA'))
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--policy', type=Path, default=DEFAULT_POLICY)
    args = parser.parse_args()
    if args.action == 'build':
        build(args.source_root, args.output, args.policy)
    else:
        verify(args.output, args.policy)
