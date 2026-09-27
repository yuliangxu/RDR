# CelebA: final Agent 3 workflow

The current results and eight figures are in [agent3.md](agent3.md). The primary experiment uses 60,000 training, 20,000 validation, and 18,000 test images per distribution. The test consists of two historically separate 9,000-image sets; this origin is retained as provenance, not a second evaluation role. All scientific outputs belong under `/cwork`.

Two reproduction paths are provided. The first reproduces the published numbers exactly from the archived learned predictions and feature caches. The second regenerates images/features, trains the models, scores the test, and builds a new report using the same fixed split manifests. Fresh training is not a claim of bitwise equality across hardware or library versions.

## Published tables and figures

From `/hpc/home/yx306/RDR`:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 python3 -B experiments/JRSSB/CELEBA_single_test.py
python3 -B experiments/JRSSB/CELEBA_package_paper.py
```

Default results: `/cwork/yx306/RDR/JRSSB/ddim_diffusion_stylegan2_equal_fid/rdr/real_generator/single_test/`. Default paper ZIP: the same experiment root's `paper/agent3_paper_2026-09-13.zip`. `--output-dir` and `--output` can select other `/cwork` locations.

## Feature and pixel RDR calibration intervals

[CELEBA_feature_ci.py](CELEBA_feature_ci.py) and [CELEBA_pixel_ci.py](CELEBA_pixel_ci.py) apply the existing [C.1 and C.2 algorithms](CI/agent_CI.md) to the frozen models for real P versus Q_L and Q_U. The current calibration set is exactly the primary test set: **18,000 images per distribution**, with the same P images in both contrasts. These rows are disjoint from the 60,000 training and 20,000 validation images, identities, and generated seeds. The 20 equal-width score bins and fitted models are unchanged.

```bash
python3 -B experiments/JRSSB/CELEBA_feature_ci.py --calibration-mode full-test
python3 -B experiments/JRSSB/CELEBA_pixel_ci.py --calibration-mode full-test
```

See the [feature report](/cwork/yx306/RDR/JRSSB/CI/celeba_feature_full_test_20260918/report.md) and [pixel report](/cwork/yx306/RDR/JRSSB/CI/celeba_pixel_full_test_20260918/report.md), including all cell/sample intervals, manifests, provenance, and PNG/PDF figures. Use a fresh `--output-dir` for a rerun. C.2 provides simultaneous intervals for population score-bin averages under its fixed-map and independent-sampling assumptions; displaying those intervals on the same test/calibration observations needs no separate query holdout. C.1 remains marginal for a fixed cell. These intervals are distinct from the Hellinger-divergence bootstrap intervals in the primary report.

Historical design use of half the test pool is recorded in the manifests. These nominal intervals do not adjust for earlier protocol selection or repeated photos within identities, and they do not measure coverage or independent calibration performance. Pixel scores equal to 0 or 2 are retained; C.1 unavailable cells are marked separately so their full-range fallbacks do not obscure C.2. The previous smaller-calibration results are preserved under `celeba_feature_20260918/` and `celeba_pixel_20260918/`.

## Whole workflow

Inputs are the raw CelebA JPEGs, the pinned Diffusion-StyleGAN2 generator and Inception weights, and the versioned split manifests already stored under `/cwork/yx306/RDR/JRSSB/`. Split manifests are dataset inputs: preparation verifies and copies their exact IDs, rather than rerunning the historical truncation search or choosing a new split. The fresh path does not use the old learned checkpoints or predictions. The generator checkpoint SHA256 is `921e72e290870affb879bc90fe5334e8cb6d5f90ff486e4d6b540036a5606745`; truncation values are 0.650 and 1.569.

Use a new work directory. The following defaults to `/cwork/yx306/RDR/JRSSB/agent3_reproduction`; `--work-root` can choose another directory. It must differ from the immutable source roots.

```bash
python3 -B experiments/JRSSB/CELEBA_workflow.py prepare
python3 -B experiments/JRSSB/CELEBA_workflow.py generate
python3 -B experiments/JRSSB/CELEBA_workflow.py features --fresh

for branch in lower upper; do
  for level in feature pixel; do
    python3 -B experiments/JRSSB/CELEBA_workflow.py train --branch "$branch" --level "$level" --fresh
    python3 -B experiments/JRSSB/CELEBA_workflow.py score --branch "$branch" --level "$level" --fresh
    for repeat in 0 1 2 3 4; do
      python3 -B experiments/JRSSB/CELEBA_workflow.py train --null --branch "$branch" --level "$level" --repeat "$repeat" --fresh
      python3 -B experiments/JRSSB/CELEBA_workflow.py score --null --branch "$branch" --level "$level" --repeat "$repeat" --fresh
    done
  done
done
for repeat in 0 1 2 3 4; do
  python3 -B experiments/JRSSB/CELEBA_workflow.py train --null --branch real --level pixel --repeat "$repeat" --fresh
  python3 -B experiments/JRSSB/CELEBA_workflow.py score --null --branch real --level pixel --repeat "$repeat" --fresh
done
python3 -B experiments/JRSSB/CELEBA_workflow.py report --fresh
```

Run generation, extraction and training on an allocated GPU. Retained `.slurm` files provide preparation and training launchers; they use `JRSSB_WORK_ROOT` and never enforce the retired FID/stabilization gates. Scoring and the final report follow training explicitly. `--allow-cpu` permits bounded checks. `--limit` is only for a separate directory containing `smoke` in its name and cannot produce the full final report.

Without `--fresh`, the workflow may reuse the original generated shards and feature caches, verifying available hashes. With `--fresh`, missing regenerated inputs cause an error. Raw image preprocessing, model architecture, numerical ratio clipping and estimator orientation remain fixed. Training sees only training/validation images and features; test inference is a separate command.

Generated nulls use their exact recorded folds, 60,000/10,000/5,000 observations per fold, and the scaler fitted on the lower and upper primary training features. Real pixel nulls use 60,000/19,000/9,000 observations per fold. All 25 fits remain learned-only. No constant-ratio substitution or null gate is applied.

## Reading the code

Start with [celeba_workflow/CELEBA_cli.py](celeba_workflow/CELEBA_cli.py): it dispatches the six stages in the commands above. Each stage has its own module. [CELEBA_workflow.py](CELEBA_workflow.py) is the command-line entrypoint; all Python modules, stage launchers, and Slurm files now use the `CELEBA_` prefix. Stage names and arguments are unchanged. Python retains its required `__init__.py` package file.

| Reading order | Module | Responsibility |
| --- | --- | --- |
| 1 | [CELEBA_paths.py](celeba_workflow/CELEBA_paths.py) | Configuration directory, immutable source roots, and default `/cwork` output root. |
| 2 | [CELEBA_prepare.py](celeba_workflow/CELEBA_prepare.py) | Verify primary splits, generated null folds, and real null folds; write `inputs.json`. |
| 3 | [CELEBA_data.py](celeba_workflow/CELEBA_data.py) | Verify input hashes, resolve image shards, and gather images/features in manifest order. |
| 4 | [CELEBA_generation.py](celeba_workflow/CELEBA_generation.py) | Generate images from recorded seeds and extract Inception features. |
| 5 | [CELEBA_fitting.py](celeba_workflow/CELEBA_fitting.py) | Resolve each experiment's settings; fit training scalers/models; restore checkpoints for test scoring. |
| 6 | [CELEBA_report.py](celeba_workflow/CELEBA_report.py) | Compute primary estimates, paired bootstrap intervals, FID, null diagnostics, and eight figures. |

`prepare()` and `report()` provide short overviews, with named helpers for their individual steps. In `CELEBA_fitting.py`, read `experiment()` for the model settings and seeds, then `train()` and `score()` for the separate fitting and evaluation paths. Fold A is the P sample and fold B is the Q sample; null experiments draw both folds from the same source.

The flow of files under the work root is:

```text
prepare  -> manifests/ + inputs.json
generate -> images/
features -> features/
train    -> models/{primary,null}/.../checkpoint.pt + loss_history.csv
score    -> models/{primary,null}/.../test_predictions.csv + test_metrics.json
report   -> report/agent3.md + tables, figures, and result.json
```

The exact reproduction of the published artifacts remains in [CELEBA_single_test.py](CELEBA_single_test.py). Its `main()` verifies split/checkpoint provenance and assembles scores; `compute_test_fid()`, `summarize_nulls()`, `render_figures()`, and `write_report()` handle the subsequent steps. [CELEBA_package_paper.py](CELEBA_package_paper.py) creates the portable paper ZIP.

Shared scientific implementations:

| File | Contents |
| --- | --- |
| [CELEBA_rdr.py](CELEBA_rdr.py) | Ratio models, training loss, optimization loop, divergence estimates. |
| [CELEBA_final_test.py](CELEBA_final_test.py) | Paired bootstrap and percentile intervals; its historical name does not introduce another test split. |
| [CELEBA_selection.py](CELEBA_selection.py) | Feature moments and FID calculation; no model-selection gate. |
| [CELEBA_plots.py](CELEBA_plots.py) | Figure layouts, score bins, and deterministic image selection. |
| [CELEBA_null_data.py](CELEBA_null_data.py) | Archived learned null prediction locations and loading. |
| [CELEBA_data.py](CELEBA_data.py), [CELEBA_split_data.py](CELEBA_split_data.py) | Preprocessing, hashing, and real-identity fold reconstruction. |
| [CELEBA_agent3.py](CELEBA_agent3.py), [CELEBA_inception.py](CELEBA_inception.py) | Generator assets, image loading, and Inception extraction. |

[configs/](configs/) holds the fixed generator, preprocessing, and primary/null training settings. `CELEBA_workflow_report.py` is a compatibility import for the new report module.

The CNN definition in `utils/DRE_func.py` and the pinned external generator checkout remain dependencies outside this folder. Install a compatible environment using `requirements.txt`; `requirements.lock.txt` records the environment used to verify the cleanup. No automated downloading or full training is triggered by importing a module.

## Checks

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python3 -B -m pytest -q -p no:cacheprovider experiments/JRSSB/tests/CELEBA_test.py
```

Tests cover loss/report agreement, pooling, paired bootstrap, FID, score-bin selection, pixel scaling, fresh-shard handling and primary architectures. The cleanup audit and bounded workflow logs are saved under `/cwork/yx306/RDR/JRSSB/`; full-scale GPU retraining is a separate verification step.
