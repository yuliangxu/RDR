"""Full-pool selection inputs, anchored to the completed expanded FID study."""
import csv
import io
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.CelebA.selection_data import (
    SOURCES, _audit_sources, _assert_disjoint, _fingerprint, _json,
    _receipt, _resolve, sha256_file,
)


def audit_expanded_sources(source_root, cfg):
    # Reuse the historical lock chain, then select all full-manifest rows.
    original, inputs, receipts = _audit_sources(source_root, cfg)
    expanded = Path(cfg['expanded_fid_root'])
    plan_path = expanded / 'plan.json'
    plan = _json(plan_path)
    receipts[str(plan_path)] = _receipt(plan_path, (expanded / 'plan.sha256').read_text().strip())
    pair_path = source_root / 'locks/generator_pair_lock.json'
    if plan['pair_lock_sha256'] != receipts[str(pair_path)]['sha256']:
        raise ValueError('Expanded and original generator pairs differ')
    for origin in original:
        for source in SOURCES:
            key = f'{source}_{origin}'
            item = inputs[key]
            full = pd.read_csv(item['input_manifest']['path'])
            full = full.assign(feature_row=np.arange(len(full)), feature_key=key,
                               original_role=origin, original_selected_row=np.arange(len(full)))
            original[origin][source] = full
            item['selected_count'] = len(full)
            item['selected_fingerprint'] = item['input_fingerprint']
            if source == 'real':
                item.pop('pixel_path', None)
                item.pop('pixel_sha256', None)
                item['pixel_from_jpeg'] = True
    for role in ('validation', 'final'):
        for branch in ('lower', 'upper'):
            path = expanded / 'locks' / f'extra_{role}_{branch}.json'
            receipts[str(path)] = _receipt(path)
            record = _json(path)
            if (record['status'] != 'generated_and_validated' or record['count'] != 20000
                    or record['role'] != role or record['branch'] != branch
                    or record['plan_sha256'] != receipts[str(plan_path)]['sha256']
                    or record['pair_lock_sha256'] != plan['pair_lock_sha256']):
                raise ValueError('Expanded pool completion/lineage mismatch')
            manifest = Path(record['manifest_path'])
            receipts[str(manifest)] = _receipt(manifest, record['manifest_sha256'])
            frame = pd.read_csv(manifest)
            if len(frame) != 20000 or _fingerprint(frame) != record['manifest_fingerprint']:
                raise ValueError('Expanded manifest count/fingerprint mismatch')
            expected_seeds = np.arange(record['seed_start'], record['seed_end_inclusive'] + 1)
            if not np.array_equal(frame.latent_seed, expected_seeds):
                raise ValueError('Expanded seed order differs from locked feature order')
            key = f'{branch}_extra_{role}'
            origin = 'validation' if role == 'validation' else 'test'
            expected = inputs[f'{branch}_{origin}']['expected_generator']
            if not np.allclose(frame.truncation_psi, expected['truncation_psi']):
                raise ValueError('Expanded branch truncation differs')
            if len(record['shards']) != 20:
                raise ValueError('Incomplete expanded shard inventory')
            # Final images/features remain unopened. Record expected hashes now;
            # development images are verified when gathered into training inputs.
            shard_hashes = {}
            for shard in record['shards']:
                image_path = str(Path(shard['path']) / 'images.pt')
                shard_hashes[image_path] = shard['image_sha256']
            if set(frame.tensor_path) != set(shard_hashes):
                raise ValueError('Expanded manifest does not match shard inventory')
            frame = frame.assign(feature_key=key, feature_row=np.arange(len(frame)),
                                 original_role=f'extra_{role}', original_selected_row=np.arange(len(frame)))
            inputs[key] = dict(feature_path=record['feature_cache_path'],
                               feature_sha256=record['feature_cache_sha256'],
                               input_fingerprint=record['manifest_fingerprint'],
                               selected_fingerprint=record['manifest_fingerprint'],
                               input_count=20000, selected_count=20000,
                               expected_generator=expected, shard_sha256=shard_hashes)
            original[origin][branch] = pd.concat([original[origin][branch], frame], ignore_index=True)
    for source in SOURCES:
        actual = [len(original['train'][source]), len(original['validation'][source]),
                  len(original['design'][source]) + len(original['test'][source])]
        expected = [122984, 39786, 39829] if source == 'real' else [120000, 40000, 40000]
        if actual != expected:
            raise ValueError(f'Expanded role counts disagree for {source}: {actual}')
    _assert_disjoint(original)
    return original, inputs, receipts


def gather_real_pixels(frame, target, receipts):
    """Canonical PIL crop/resize; retain a SHA256 ledger of every raw JPEG."""
    import hashlib
    import torch
    from PIL import Image
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize((64, 64), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.ToTensor(),
    ])
    ledger = Path(target.filename).parent / 'real.raw_inputs.csv'
    temporary = ledger.with_suffix('.csv.partial')
    with temporary.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['source_id', 'real_path', 'sha256'])
        for row in frame.itertuples():
            path = Path(row.real_path)
            raw = path.read_bytes()
            with Image.open(io.BytesIO(raw)) as image:
                values = (255 * transform(image.convert('RGB'))).round().to(torch.uint8)
            target[row.Index] = values.numpy()
            writer.writerow([row.source_id, str(path), hashlib.sha256(raw).hexdigest()])
            if (row.Index + 1) % 20000 == 0:
                print(f'Real pixels: {row.Index + 1:,}/{len(frame):,}', flush=True)
    temporary.replace(ledger)
    receipts[str(ledger)] = {**_receipt(ledger), 'hash_basis': 'raw_JPEG_contents_observed_at_preparation'}
