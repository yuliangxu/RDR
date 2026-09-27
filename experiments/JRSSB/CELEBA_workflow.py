#!/usr/bin/env python3
"""Command-line entrypoint for the final CelebA Agent 3 workflow.

Implementation and reading order: celeba_workflow/ and README.md.
The stage names and arguments are preserved for existing Slurm launchers.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/agent3-matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from celeba_workflow.CELEBA_cli import main

# Preserve imports used by existing notebooks and the scientific contract tests.
from celeba_workflow.CELEBA_data import ManifestImages, feature_values, image_shard
from celeba_workflow.CELEBA_fitting import model_and_optimizer
from celeba_workflow.CELEBA_paths import HERE

if __name__ == "__main__":
    main()
