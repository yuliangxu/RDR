#!/usr/bin/env python3
"""Acquire the six inputs recorded by the completed MNIST comparison.

MNIST comes from the public torchvision mirror; the pretrained generators come
from csinva/gan-vae-pretrained-pytorch. SHA-256 values are those of the raw inputs
in results/MNIST/selected_comparison/protocol.json. Downloaded MNIST archives are
decompressed before verification. Existing files are never overwritten.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
from urllib.error import URLError
from urllib.request import urlopen


@dataclass(frozen=True)
class Input:
    relative: str
    url: str
    sha256: str
    compressed: bool = False


MNIST_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
GENERATOR_URL = "https://raw.githubusercontent.com/csinva/gan-vae-pretrained-pytorch/master/"
INPUTS = (
    Input("MNIST/raw/train-images-idx3-ubyte", MNIST_URL + "train-images-idx3-ubyte.gz",
          "ba891046e6505d7aadcbbe25680a0738ad16aec93bde7f9b65e87a2fc25776db", True),
    Input("MNIST/raw/train-labels-idx1-ubyte", MNIST_URL + "train-labels-idx1-ubyte.gz",
          "65a50cbbf4e906d70832878ad85ccda5333a97f0f4c3dd2ef09a8a9eef7101c5", True),
    Input("MNIST/raw/t10k-images-idx3-ubyte", MNIST_URL + "t10k-images-idx3-ubyte.gz",
          "0fa7898d509279e482958e8ce81c8e77db3f2f8254e26661ceb7762c4d494ce7", True),
    Input("MNIST/raw/t10k-labels-idx1-ubyte", MNIST_URL + "t10k-labels-idx1-ubyte.gz",
          "ff7bcfd416de33731a308c3f266cc351222c34898ecbeaf847f06e48f7ec33f2", True),
    Input("mnist_dcgan/netG_epoch_99.pth", GENERATOR_URL + "mnist_dcgan/weights/netG_epoch_99.pth",
          "d61db3ef26f469f29c1725b16c2be82cecd1a53fc53bcdda9f8789595a3d65f3"),
    Input("mnist_vae/vae_epoch_25.pth", GENERATOR_URL + "mnist_vae/weights/vae_epoch_25.pth",
          "233916a8095759cf732f4dc19c381bc8de03d97040e68f4252a257a5502a9c70"),
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path, expected):
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: expected {expected}, got {actual}")
    return actual


def acquire(data_root, item, verify_only=False):
    """Return an input record after verification, reusing correct existing files."""
    target = Path(data_root) / item.relative
    if target.exists():
        verify(target, item.sha256)
        return {"path": item.relative, "sha256": item.sha256, "status": "verified"}
    if verify_only:
        raise FileNotFoundError(f"Missing input: {target}")
    if not item.url.startswith("https://"):
        raise ValueError("Input acquisition requires an HTTPS URL")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix="." + target.name + ".",
                                         suffix=".partial", delete=False) as output:
            temporary = Path(output.name)
            try:
                response = urlopen(item.url, timeout=60)
            except URLError as error:
                raise OSError(f"Cannot download {target} from {item.url}: {error}") from error
            with response:
                stream = gzip.GzipFile(fileobj=response) if item.compressed else response
                try:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        output.write(block)
                finally:
                    if item.compressed:
                        stream.close()
            output.flush()
            os.fsync(output.fileno())
        verify(temporary, item.sha256)
        # Same-directory hard linking installs the complete verified file
        # atomically, with exclusive creation instead of overwriting a racer.
        try:
            os.link(temporary, target)
            status = "downloaded"
        except FileExistsError:
            verify(target, item.sha256)
            status = "verified"
        return {"path": item.relative, "sha256": item.sha256, "status": status}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def acquire_all(data_root, verify_only=False):
    # Reject any conflicting existing input before starting a download.
    for item in INPUTS:
        target = Path(data_root) / item.relative
        if target.exists():
            verify(target, item.sha256)
    return [acquire(data_root, item, verify_only) for item in INPUTS]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Directory containing MNIST/raw, mnist_dcgan and mnist_vae")
    parser.add_argument("--verify-only", action="store_true",
                        help="Verify all six existing files without writing or downloading")
    args = parser.parse_args()
    try:
        records = acquire_all(args.data_root, args.verify_only)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Input preparation failed: {error}\n")
    print(json.dumps({"data_root": str(args.data_root), "inputs": records}, indent=2))


if __name__ == "__main__":
    main()
