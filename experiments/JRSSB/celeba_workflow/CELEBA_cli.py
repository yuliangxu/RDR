"""Run one stage: prepare, generate, features, train, score, or report.

All scientific outputs are written under /cwork. Use --help for stage options.
"""

import argparse
from pathlib import Path
import torch
from .CELEBA_paths import SOURCE, REFERENCE, DEFAULT_WORK
from .CELEBA_data import read_json, verify
from .CELEBA_prepare import prepare
from .CELEBA_generation import generate, extract_features
from .CELEBA_fitting import train, score


def main(argv=None):
    """Validate command-line options and dispatch one explicit workflow stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=["prepare", "generate", "features", "train", "score", "report"]
    )
    parser.add_argument("--source-root", type=Path, default=SOURCE)
    parser.add_argument("--reference-root", type=Path, default=REFERENCE)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--branch", choices=["real", "lower", "upper"])
    parser.add_argument("--level", choices=["feature", "pixel"], default="feature")
    parser.add_argument("--null", action="store_true")
    parser.add_argument("--repeat", type=int, choices=range(5), default=0)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--fresh", action="store_true", help="Require newly regenerated images/features"
    )
    parser.add_argument(
        "--limit", type=int, help="Smoke test only; use a separate work root"
    )
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args(argv)
    args.work_root = args.work_root.resolve()
    if not str(args.work_root).startswith("/cwork/"):
        raise ValueError("Scientific outputs must be saved under /cwork")
    if args.work_root in (args.source_root.resolve(), args.reference_root.resolve()):
        raise ValueError("Use a fresh work root; historical sources are read-only")
    if args.limit and (args.limit < 2 or "smoke" not in str(args.work_root)):
        raise ValueError(
            "--limit requires >=2 observations and a separate smoke work root"
        )
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if (
        args.stage in ("generate", "features", "train", "score")
        and args.device.type == "cpu"
        and not args.allow_cpu
    ):
        raise RuntimeError("CUDA required unless --allow-cpu is specified")
    if args.stage == "prepare":
        return prepare(args)
    spec = read_json(args.work_root / "inputs.json")
    for path, digest in spec.get("manifest_sha256", {}).items():
        verify(path, digest)
    if args.stage == "generate":
        return generate(args, spec)
    if args.stage == "features":
        return extract_features(args, spec)
    if args.stage == "report":
        from .CELEBA_report import report

        return report(args, spec)
    if args.branch is None:
        parser.error("--branch is required")
    if args.branch == "real" and not args.null:
        parser.error("--branch real is only valid with --null")
    if args.stage == "train":
        return train(args, spec)
    return score(args, spec)
