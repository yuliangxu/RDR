# CelebA experiments

This is the dataset-level inventory for the paper repository. **All experiment
families below are pending paper inclusion.** Listing an existing experiment
does not select it for the paper or certify a new reproduction run.

The planned code location is `experiments/CelebA/`. The links below describe the
current checkout; experiment entrypoints remain in place; dataset helpers have moved out of `utils`. Dataset transforms,
generator adapters, split preparation, and experiment reports belong with these
experiments. Reusable RDR methods and calibration algorithms belong in `utils/`.

## Experiment inventory

| Family | Purpose | Current entrypoint | Paper inclusion |
| --- | --- | --- | --- |
| Fixed generator truncation comparison | Compare FID and feature/pixel RDR for two distributions from a frozen generator. | [CELEBA_workflow.py](../experiments/JRSSB/CELEBA_workflow.py) | Pending |
| Learned same-source nulls | Inspect auxiliary null fits for generated and real samples. | [Workflow null stages](../experiments/JRSSB/README.md#whole-workflow) | Pending |
| Feature/pixel calibration intervals | Attach C.1/C.2 cell-average intervals to frozen-model scores. | [CELEBA_feature_ci.py](../experiments/JRSSB/CELEBA_feature_ci.py), [CELEBA_pixel_ci.py](../experiments/JRSSB/CELEBA_pixel_ci.py) | Pending |
| Earlier real-versus-DDIM comparisons | Compare real and DDIM images in pixel and embedding spaces. | [CelebA_ddim.py](../experiments/CelebA_ddim.py), [embedding script](../experiments/CelebA_ddim_inception_embeddings.py) | Pending |
| Earlier real-halves nulls | Inspect same-source comparisons in pixel and embedding spaces. | [CelebA_two_halves.py](../experiments/CelebA_two_halves.py), [embedding script](../experiments/CelebA_two_halves_inception_embeddings.py) | Pending |

The existing [Agent 3 report](../experiments/JRSSB/agent3.md) records the fixed
generator comparison and learned auxiliary nulls. The [workflow guide](../experiments/JRSSB/README.md)
provides current execution commands and links to the calibration reports.
These reports depend on saved artifacts outside Git; this inventory does not
replace or update their scientific results.

## Split roles and interpretation

The primary workflow keeps training, validation/early stopping, and test roles
separate, with recorded image IDs, real identities, and generated seeds. Its current
test pool combines historically separate design and final-test rows. This is a
**retrospective held-out reanalysis**, not a newly sealed confirmatory experiment.
Auxiliary null fits have their own recorded folds and remain learned estimators.

The feature/pixel CI workflows use the primary test pool for calibration with
frozen models. Their target is the population average RDR within a score cell,
not individual-point RDR. Earlier protocol selection and repeated photos within
identities limit these to nominal retrospective image-level diagnostics; they
do not establish empirical coverage. Earlier DDIM/null scripts have separate
protocols that must be audited independently if selected.

## Reproduction paths

The [published-artifact commands](../experiments/JRSSB/README.md#published-tables-and-figures)
use [CELEBA_single_test.py](../experiments/JRSSB/CELEBA_single_test.py) to rebuild
the report from saved learned predictions and feature caches. The
[paper packager](../experiments/JRSSB/CELEBA_package_paper.py) bundles the report,
figures, and linked tables; it is not a dataset or training-checkpoint archive.

The [whole-workflow commands](../experiments/JRSSB/README.md#whole-workflow) cover
`prepare`, `generate`, `features`, `train`, `score`, and `report`. The `--fresh`
path regenerates required images/features and fits new RDR models. Preparation
uses the recorded split manifests; it does not repeat historical split or
truncation selection. Fresh training is distinct from saved-score replay and is
not a promise of bitwise equality. No reproduction was run for this inventory.

## Work needed for a portable paper release

- Select families before consolidating entrypoints, configuration, and reporting.
- Provide acquisition instructions and manifests for raw CelebA images, exact
  splits, the pinned Diffusion-GAN checkout/checkpoint, and Inception weights.
  These currently depend on external `/cwork` and HPC data locations.
- Make input/output roots configurable and replace `/cwork`-only output guards.
  Update sibling imports, package paths, Slurm launchers, and report links together.
- Consolidate the relocated [CelebA helpers](../experiments/CelebA/helpers.py) and the
  [DDIM sampler](../experiments/CelebA/ddim.py) within experiment-specific code;
  isolate optional image-analysis dependencies from shared RDR imports.
- Use the shared calibration module while preserving the frozen-model
  cell-average target. Keep dataset preprocessing and generator logic local.
- Pin and document required environments using the existing
  [requirements](../experiments/JRSSB/requirements.txt) as an input to the review.
- Choose small figures/tables for optional `results/CelebA/`; larger caches,
  datasets, and checkpoints need separate distribution and provenance records.
