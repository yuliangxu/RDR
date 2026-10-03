# Finalized expanded CelebA experiment

Read [docs/CelebA.md](../../docs/CelebA.md) for scientific results, exact split
counts, interpretation and the experiment inventory. The
[paper package](../../results/CelebA/README.md) contains 32 final PNG/PDF pairs,
20 selected-model assessments and 25 learned same-source null assessments.

## Pipeline

1. [Expand fixed generator pools and compute FID](expanded_fid.md).
2. [Select loss and output activation](expanded_selection.md), using
   [expanded_selection_config.json](expanded_selection_config.json).
   Keep feature/pixel architectures fixed; 319/320 fits completed, with one
   documented Hellinger numerical failure. Accepted settings are JS with
   slopes 0.5 (feature) and 2 (pixel), for both Q pairs.
3. [Evaluate accepted checkpoints](final_evaluation.py) and
   [fit the five learned null families](null_experiment.py).
4. [Merge the final roles](merged_test.py) into one independent-of-training
   test pool; [render three-digit samples and null histograms](publication_figures.py).
5. [Regress stable log(2−RDR) on annotations](attribute_regression.py), then
   [assess interactions and response sensitivity](attribute_interactions.py)
   with [publication figures](attribute_interaction_figures.py).
6. [Export and verify final artifacts](package_paper.py), using the pinned
   [file/hash policy](paper_files.json).

## Commands and prerequisites

Exact saved-run replay, fresh-scoring, null fitting and attribute commands are
in [the scientific summary](../../docs/CelebA.md#reproducible-expanded-data-pipeline)
and [REPRODUCTION.md](../../results/CelebA/REPRODUCTION.md). Use fresh output
directories for new fits. Archived source trees retain the versions used for
the original results. The former calibration/evaluation outputs are required
inputs to merged replay; their superseded plots are omitted from this package.

Python, PyTorch, NumPy, SciPy, pandas, matplotlib, pytorch-fid and CelebA images,
metadata, generator/Inception weights, manifests and arrays are required for
full replay. Attribute OLS uses statsmodels 0.14.5; the interaction study uses
InterpretML 0.7.8. Recorded versions and external asset hashes are linked in
REPRODUCTION.md. Slurm wrappers and several guarded paths are cluster-specific;
the package is not a tested fresh-machine installation recipe.

Offline evidence verification needs only the Python standard library:

```bash
python3 -B experiments/CelebA/package_paper.py verify
```

Focused numerical/pipeline checks:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg \
python3 -B -m pytest -q -p no:cacheprovider tests/test_celeba_*.py
```

## Dependency and archive boundary

`selection_data.py`, `selection_config.json` and `final_figures.py` retain
shared preparation, training configuration and verification/rendering helpers
used by the expanded workflow and its tests. Four support modules and the
fixed generator configuration remain in [JRSSB](../JRSSB/README.md).
Frozen source archives are provenance for exact replay; they are not extra
paper experiments. Unused DDIM/real-halves and smaller-data entrypoints were
removed only after a verified complete snapshot was published to
[RDR-working](https://github.com/yuliangxu/RDR-working/tree/celeba-working-20261003).
