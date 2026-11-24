"""Recover the generated-only ranked montages from the frozen CelebA evaluation."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def archive_montages(run: Path, reference: Path, groups: dict) -> dict:
    """Verify archived inputs and return the native RGB montages in manifest order."""
    roles = ("test_calibration", "test_evaluation")
    branches = ("lower", "upper")
    representations = ("feature", "pixel")
    expected = {(rep, branch, target) for rep in representations for branch in branches
                for target in range(3)}
    if set(groups) != expected:
        raise ValueError("Expected the twelve generated ranked groups")

    recorded = json.loads((reference / "protocol.json").read_text())["input_hashes"]
    anchor = Path("data/arrays/pixel/test_calibration/lower.npy")
    original, = (Path(p).parents[len(anchor.parts) - 1] for p in recorded
                 if Path(p).parts[-len(anchor.parts):] == anchor.parts)

    def verified(relative):
        path = run / relative
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(4 << 20), b""):
                hasher.update(block)
        if hasher.hexdigest() != recorded[str(original / relative)]:
            raise ValueError(f"Frozen input hash mismatch: {relative}")
        return path

    montages = {}
    for branch in branches:
        images, identities = [], []
        for role in roles:
            path = verified(Path("data/arrays/pixel") / role / f"{branch}.npy")
            bank = np.load(path, mmap_mode="r", allow_pickle=False)
            with verified(Path("data/manifests") / f"{role}_{branch}.csv").open(newline="") as f:
                ids = [row["source_id"] for row in csv.DictReader(f)]
            if bank.shape != (len(ids), 3, 64, 64) or bank.dtype != np.uint8:
                raise ValueError(f"Unexpected generated bank shape or dtype for {branch}/{role}")
            images.append(bank)
            identities.extend((role, source_id) for source_id in ids)

        for rep in representations:
            pattern = f"evaluation/{rep}/{branch}/*/repeat_00/predictions.npz"
            prediction, = (Path(p).relative_to(original) for p in recorded
                           if original in Path(p).parents and Path(p).match(pattern))
            with np.load(verified(prediction), allow_pickle=False) as scores:
                parts = [scores[f"{role}_q"] for role in roles]
            if any(part.shape != (len(bank),) for part, bank in zip(parts, images)):
                raise ValueError(f"Unexpected generated score counts for {rep}/{branch}")
            values = np.concatenate(parts)
            for target in range(3):
                key = rep, branch, target
                rows = groups[key]
                if [int(r["display_rank"]) for r in rows] != list(range(40)):
                    raise ValueError(f"Incorrect display ranks for {key}")
                canvas = np.full((259, 649, 3), 255, dtype=np.uint8)
                for rank, row in enumerate(rows):
                    index = int(row["test_row"])
                    if (row["source"] != branch or not 0 <= index < len(values)
                            or identities[index] != (row["original_final_role"], row["source_id"])
                            or float(values[index]) != float(row["rdr"])):
                        raise ValueError(f"Frozen generated row disagrees with {key}, rank {rank}")
                    role_index = int(index >= len(images[0]))
                    offset = len(images[0]) if role_index else 0
                    y, x = divmod(rank, 10)
                    canvas[y*65:y*65+64, x*65:x*65+64] = images[role_index][index-offset].transpose(1, 2, 0)
                montages[key] = canvas
    return montages
