# AGP

This is an inventory for repository cleanup, not the final list of paper experiments.
**Every family below is pending inclusion.** Shared utility dependencies and dataset helpers have been relocated,
and no model fitting or replay has been run as part of this inventory.

The planned experiment location is `experiments/AGP/`, which already contains
most AGP workflows. Links below refer to current sources and historical notes.

## Experiment families

| Family | Purpose | Current source |
| --- | --- | --- |
| Preprocessing and generator history | Prepare microbiome observations and train/use the historical ICFM generator. | [Preprocessing](../experiments/AGP/AGP1_data_preprocessing.py), [ICFM workflow](../experiments/AGP/AGP2_ICFM.py), [microbiome network](../experiments/AGP/microbiome_unet.py) |
| Same-source null | Compare random halves of real observations to diagnose departures from the null RDR of one. | [AGP_two_halves.py](../experiments/AGP/AGP_two_halves.py) |
| Original real-versus-ICFM fit | Fit a bounded RDR model with audited subjects and separate calibration/query subsets. | [Fit and CI](../experiments/AGP/AGP_ICFM_ci.py), [data audit](../experiments/AGP/AGP_ci_data.py) |
| Frozen full-test replay | Reuse the original checkpoint and combine its former calibration/query subsets for cell-average intervals. | [AGP_ICFM_test_ci.py](../experiments/AGP/AGP_ICFM_test_ci.py) |
| Revised training/validation allocation | Refit after moving fit-pool observations into training while preserving the full test cohort. | [7:1:2-policy fit](../experiments/AGP/AGP_ICFM_712_ci.py), [allocation helper](../experiments/AGP/AGP_712_data.py) |
| Adjacent-cell merging | Recompute intervals on selected neighboring score regions with correction over the candidate family. | [AGP_ICFM_merged_ci.py](../experiments/AGP/AGP_ICFM_merged_ci.py) |
| Compositional representations | Compare raw/closed abundances, ILR, learned log-contrasts and phylogenetic ILR, including smoothing sensitivity. | [Runner](../experiments/AGP/AGP_composition_diagnostics.py), [report](../experiments/AGP/AGP_composition_report.py), [phylogenetic basis](../experiments/AGP/AGP_phylo_basis.py) |
| Architecture comparison | Compare patience controls and wide, deep and residual MLPs on the same ILR representation and archived splits. | [Runner](../experiments/AGP/AGP_architecture_diagnostics.py), [report](../experiments/AGP/AGP_architecture_report.py) |
| Residual-model CI extensions | Inspect a frozen residual model using equal-width/adaptive cells and bounds on true within-cell variance. | [Equal-width](../experiments/AGP/AGP_residual_equal_width_ci.py), [adaptive](../experiments/AGP/AGP_residual_adaptive_ci.py), [variance](../experiments/AGP/AGP_residual_variance_ci.py) |

The existing [AGP workflow notes](../experiments/AGP/README.md) provide detailed
historical commands, selections and artifact links. These workflows are distinct
protocols; their fits and sample allocations should not be silently combined.

## Targets and split roles

P denotes retained real AGP observations and Q denotes archived ICFM observations.
The original representation has 614 taxa. The midpoint target is
`r0(x) = 2p(x)/(p(x)+q(x))`. Separate empirical P/Q means retain that target when
the two sample sizes differ.

The current keep-all allocation has the following counts:

| Role | P observations | Q observations |
| --- | ---: | ---: |
| Training | 9,162 | 7,310 |
| Validation | 1,281 | 1,044 |
| Test = calibration | 2,002 | 2,089 |

The real allocation is approximately 74/10/16 because the generator-unseen test
pool is fixed; Q is approximately 70/10/20. The audit excludes 609 real test
observations whose subjects appear in generator training. Subject separation
does not remove repeated observations within a subject.

Checkpoints and architecture/representation selection use validation data.
Training and validation calibration curves are descriptive. The test cohort has
been inspected across successive diagnostics, so comparisons remain retrospective.

C.1/C.2 intervals target true **cell-average RDR**, not each individual RDR value.
Different fitted networks induce different input-space cells even when numeric
score cutoffs match. Merged regions and elementary cells also have different
targets and multiplicity corrections. An interval of `[1,1]` for the entire
score range gives no evidence of regional calibration.

AGP intervals retain a nominal interpretation: within-subject dependence,
historical whole-table preprocessing and incomplete generator provenance limit
IID coverage claims. Lower aggregate error can coexist with larger middle-score
gaps; retain cell masses and conditional gaps alongside aggregate metrics.

## Existing results: local artifact pointers

These are **local research paths, not portable GitHub links**. Each listed folder
currently has a report and a saved `result.json` reporting `complete`:

- `/cwork/yx306/RDR/JRSSB/CI/agp_icfm_712_20260920/report.md`
- `/cwork/yx306/RDR/JRSSB/CI/agp_composition_20260922/report.md`
- `/cwork/yx306/RDR/JRSSB/CI/agp_architecture_20260925/report.md`
- `/cwork/yx306/RDR/JRSSB/CI/agp_residual_equal_width_ci_20260925/report.md`

These saved statuses were read during inventory. Full artifact hashes and fresh
reproduction were not checked. Historical and newer diagnostic outputs remain
separate; no figures or result files have been selected for publication yet.

## Work needed before publication

- Document acquisition/access and hashes for the external data root, currently
  `/hpc/group/mastatlab/yx306/AGP/data`. The CI audit requires
  `yx1_xtrain_raw.csv`, `yx1_xtest_raw.csv`, `yx1_icfm_recon_train.csv`,
  `yx1_icfm_recon_test.csv`, `yx1_abundance.csv`, `yx2_metadata.csv` and
  `icfm_checkpoint.pth`; phylogenetic comparisons need additional tree metadata.
- Separate reproduction from archived generator samples from end-to-end generator
  training. The historical ICFM script changes directory to another checkout and
  imports external lab code and legacy utilities; those dependencies need auditing.
- Keep using the shared [CI algorithms](../utils/calibration.py); AGP data and
  composition logic remain in the experiment folder.
- Replace personal output-prefix restrictions and make data/checkpoint locations
  configurable. Update source snapshots and path-dependent provenance checks too.
- Record tested dependencies and environment, including PyTorch, NumPy, SciPy,
  pandas, scikit-learn and plotting packages; audit legacy Keras/generator imports.
- Preserve row identities, subject assignments, taxon order, transformations,
  checkpoint selection and source hashes when consolidating workflows.
- Add portable reproduction commands and selected small `results/` artifacts
  after inclusion decisions; retain the existing sources until migration is verified.
