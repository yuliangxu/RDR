# CelebA reproduction scope

## What works from this checkout

All retained reports, tables and 32 current PNG/PDF figure pairs can be read without
the HPC archive. Run `python3 -B experiments/CelebA/package_paper.py verify`
from the repository root to validate every exported file and frozen-source
archive. This requires only Python's standard library and no GPU or data
download. The manifest records exact upstream paths and source hashes.

The current result is [merged_test_20261003](merged_test_20261003/RESULTS.md):
former final-calibration rows followed by former final-evaluation rows form
one test pool. The same rows supply counts, CIs, neural means and Brier/Gap.
Training/development roles, selected settings, weights and score bins are
unchanged. The original split-based assessments/figures remain in the external archive
and the complete RDR-working snapshot.

This is an evidence and source package. Raw images, generated pools, Inception
arrays, per-image predictions and neural checkpoints remain in the archive;
their absence does not prevent reading or verifying the paper artifacts.
The small example-image panels are included, together with their selection
IDs, scores, seeds and bin counts.

## Recorded environment and fixed inputs

The selection protocol records Python 3.9.25, NumPy 1.26.4, pandas 2.2.3,
PyTorch 2.6.0, SciPy 1.13.1 and Matplotlib 3.9.4. The FID plan additionally
records torchvision 0.21.0 and pytorch-fid 0.3.0. These are the archived
stage-specific versions, not a complete operating-system/CUDA lock file.
The [selection protocol](model_selection_expanded_20260929/protocol.json)
and [FID plan](expanded_fid_20260929/plan.json) retain the original records.
The historical [dependency list](../../experiments/JRSSB/requirements.txt)
is an input to environment reconstruction, not a newly validated install recipe.

Required inputs for full replay/refitting are:

1. Aligned CelebA images and their identity/attribute/partition metadata,
   plus the archived role manifests. The expanded real pools have 122,984
   training, 39,786 validation and 39,829 final images, with identity-disjoint
   roles. They are the archived research partition, not a direct reuse of the
   dataset's official train/validation/test allocation. Recreating a split
   from counts alone would not reproduce these results.
2. Diffusion-GAN source at commit
   `2ca5e7aa21f07a3ff0e78966e5796c580126c14f` and the fixed
   `diffusion-stylegan2-celeba64.pkl` checkpoint with SHA-256
   `921e72e290870affb879bc90fe5334e8cb6d5f90ff486e4d6b540036a5606745`.
   The [generator provenance](expanded_fid_20260929/provenance/generator_assets.json)
   records the original repository and checkpoint download locations.
   Q_l/Q_u use truncation 0.650/1.569 with constant noise, recorded seed
   manifests and fixed image preprocessing.
3. Canonical pytorch-fid Inception weights and raw 2048-dimensional pool3
   caches. The [FID plan](expanded_fid_20260929/plan.json) records Inception
   provenance; FID uses float64 pooled moments and sample covariance.
   Feature-network standardization is fitted only to its training data.
4. For saved-model assessment, the 20 accepted JS checkpoints and final
   arrays identified by the [settings freeze](final_evaluation_expanded_20261003/freeze.json)
   and [final metadata](final_evaluation_expanded_20261003/data/evaluation_metadata.json).
   For exact null replay, retain the 25 null checkpoints/predictions and
   [A/B manifest/index provenance](null_expanded_20261003/data/null_data.json).

Large inputs are located under the original `/cwork/yx306/RDR/` and CelebA
data roots recorded in the JSON provenance. Keep those assets and all source
receipts intact. Do not substitute official dataset partitions, regenerate
different random seeds, or silently redirect a frozen protocol to new inputs.

## Reproduction stages

| Stage | Code | Inputs and expected scope |
| --- | --- | --- |
| Expand generator pools and compute FID | [Generation](../../experiments/CelebA/expanded_fid_generate.py), [FID](../../experiments/CelebA/expanded_fid_report.py), [commands](../../experiments/CelebA/expanded_fid.md) | Frozen generator, existing pools, additional seed intervals; no RDR training |
| Loss/activation selection | [Driver](../../experiments/CelebA/model_selection.py), [config](../../experiments/CelebA/expanded_selection_config.json), [protocol](../../experiments/CelebA/expanded_selection.md) | Development roles only; 320 planned fits with one recorded failure |
| Accepted-model assessment | [Driver](../../experiments/CelebA/final_evaluation.py) | Frozen accepted checkpoints; final calibration/evaluation only; no refitting |
| Selection presentation | [Renderer](../../experiments/CelebA/refresh_selection_presentation.py) | Saved validation results and accepted-settings freeze; preserves the failed cell |
| Shared CI/example-image helpers | [Renderer utilities](../../experiments/CelebA/final_figures.py) | Verification, image access and plotting helpers used by the merged pipeline |
| Null controls | [Driver](../../experiments/CelebA/null_experiment.py), [adapter](../../experiments/CelebA/null_data.py), [report](../../experiments/CelebA/null_report.py) | Fresh fits on fixed source halves with selected configurations; separate final roles |
| Current merged test and compact figures | [Driver](../../experiments/CelebA/merged_test.py) | Pool both saved final roles; recompute 20 main/25 null assessments, one count panel per main CI plot, global score-ranked examples with actual displayed ranges; no retraining |
| Current image labels and null histograms | [Renderer](../../experiments/CelebA/publication_figures.py) | [Presentation supplement](publication_figures_20261003_v2/FIGURES.md); three significant digits, identical 960 ranked images, histograms of all 25 learned null fits |
| Feature attribute associations | [Driver](../../experiments/CelebA/attribute_regression.py) | Frozen feature networks and real test features; 40 annotations joined by filename; ten joint regressions with stable log responses and identity-clustered intervals |
| Attribute interaction/response sensitivity | [Driver](../../experiments/CelebA/attribute_interactions.py), [wrapper](../../experiments/CelebA/attribute_interactions.slurm), [figures](../../experiments/CelebA/attribute_interaction_figures.py) | Parent design/stable responses, InterpretML 0.7.8; frozen identity split; 20 scenarios, 980 EBM and 20 OLS fits |
| Evidence export | [Packager](../../experiments/CelebA/package_paper.py) | One final-only build with pinned upstream hashes; 302 artifacts and 32 figure pairs |

Exact archived commands and seeds are in [docs/CelebA.md](../../docs/CelebA.md#reproducible-expanded-data-pipeline).
Use the run's frozen `source/` tree for replay after repository code changes.
Four `frozen_source.tar.gz` files preserve the original FID, selection, scoring
and null stages. Later frozen source trees are available in
[RDR-working](https://github.com/yuliangxu/RDR-working/tree/celeba-working-20261003/results/CelebA)
and the external run directories. Extracting source alone does not restore
missing arrays or checkpoints. The current pipeline is retained in this checkout.

For the current merged result, use the source saved under
`merged_test_20261003/source/`; the two earlier assessments remain its immutable
inputs. Example command with the full HPC archive available:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python3 -B /cwork/yx306/RDR/CelebA/merged_test_20261003/source/experiments/CelebA/merged_test.py \
  --main-run /cwork/yx306/RDR/CelebA/final_evaluation_expanded_20261003 \
  --null-run /cwork/yx306/RDR/CelebA/null_expanded_20261003 \
  --output /cwork/yx306/RDR/CelebA/merged_test_replay_NEW --seed 2026100301
```

For the updated image labels and null histograms, run the separate presentation
stage after the merged assessment:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python3 -B /cwork/yx306/RDR/CelebA/publication_figures_20261003_v2/source/experiments/CelebA/publication_figures.py \
  --merged /cwork/yx306/RDR/CelebA/merged_test_20261003 \
  --output /cwork/yx306/RDR/CelebA/publication_figures_replay_NEW
```

This verifies the sealed parent, preserves the exact image selections, and
recomputes histogram counts from the saved merged A/B scores. Its shaded central
95% score ranges are distribution summaries, not calibration confidence intervals.
Its six frozen source files remain under the external run
`/cwork/yx306/RDR/CelebA/publication_figures_20261003_v2/source/` and in RDR-working.

The merged run includes compact row manifests and verifies disjointness from
training/development plus real-identity separation. This remains retrospective
because historical design/test observations were previously inspected. Shared
test counts and neural means are correlated; their Gap is descriptive, not
covered by the plotted cell-average CIs. Supported mass is 1 by construction.

The attribute regression uses statsmodels 0.14.5 (recorded in its protocol),
the original `list_attr_celeba.txt`, and saved feature checkpoints/arrays.
To replay it without training:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python3 -B /cwork/yx306/RDR/CelebA/attribute_regression_expanded_20261003_v2/source/experiments/CelebA/attribute_regression.py \
  --merged /cwork/yx306/RDR/CelebA/merged_test_20261003 \
  --attributes /hpc/group/mastatlab/yx306/CelebA/celeba/list_attr_celeba.txt \
  --output /cwork/yx306/RDR/CelebA/attribute_regression_replay_NEW
```

Its archived `design.csv` and `responses/` retain the matched per-image data;
these are not duplicated in the compact repository export. The regression
report, coefficient tables, diagnostics and active source are included; the
complete frozen parent source remains in RDR-working and the external archive. Log responses
are recovered from frozen logits, including observations whose saved RDR
rounded to exactly 2; no clipping or row deletion is used.

For the interaction study, follow the commands in
[the scientific summary](../../docs/CelebA.md#attribute-interactions-and-response-sensitivity).
The original RDR models are not refitted. The new EBM explanation models use
InterpretML 0.7.8, scikit-learn 1.6.1 and the versions in
[attribute_interactions_20261003/protocol.json](attribute_interactions_20261003/protocol.json).
The driver references an isolated InterpretML installation at
`/cwork/yx306/RDR/CelebA/environments/interpret_core_0_7_8`, whose files are hashed.
On the original HPC environment it was installed with:

```bash
python3 -m pip install --no-deps \
  --target /cwork/yx306/RDR/CelebA/environments/interpret_core_0_7_8 interpret-core==0.7.8
```

Existing NumPy/pandas/scikit-learn dependencies are required; `--no-deps` is
not a fresh-machine environment recipe. The new protocol records dependency
versions, the 60/20/20 identity split, group definitions and all fit settings.
Per-task saved models and predictions remain external; exported receipts
reference those files. The publication renderer has its own source snapshot
and completion receipt linked to the numerical analysis.

Prepare new output directories for refitting or changed analyses. Fresh neural
training is not a promise of bitwise equality to archived predictions, even
with the same protocol. Saved-artifact verification is a stricter and separate
operation, tested by exact hashes.

## Remaining work for independent full reproduction

The approved experiment and paper-artifact export are finished. Before calling
this a fresh-machine training release, complete these infrastructure steps:

- Distribute the exact role manifests/index bundle and an input acquisition
  procedure that validates CelebA, generator and Inception hashes. Existing
  JSON receipts locate and describe those assets but do not contain every
  role's full image membership.
- Make data/output roots configurable across preparation, resume checks and
  launchers. Current code deliberately guards original `/cwork` locations and
  verifies frozen lineage; changing only one path is insufficient.
- Pin the remaining Python/CUDA/system dependencies and test a clean install.
  Record versions from actual execution rather than inferring them from the
  historical requirements file.
- Run a fresh-checkout integration check with the redistributed inputs, then
  validate the full replay/refit path before claiming independent reproduction.

These steps do not require selecting a different architecture, retrying the
failed Hellinger fit, or adding new scientific comparisons. Historical runners and superseded paper outputs have now been removed from
this checkout after publishing the verified complete snapshot to RDR-working.
External inputs and frozen research runs remain intact.
