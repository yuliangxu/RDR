# Shared CI methods and historical applications

The active simulation code is in [experiments/simulations](../../simulations/README.md).
Its workflow covers loss, output-activation and architecture selection followed
by convergence of a locked Hellinger configuration. See the
[simulation summary](../../../docs/simulations.md) for completed results and
reproduction commands. Sample-size convergence remains to be run.

[utils/calibration.py](../../../utils/calibration.py) implements C.1/C.2;
`calibration.py` here remains a compatibility import. Numerical tests are under
[tests](../../../tests), including fixed bins, adaptive merging and variance
bounds. [agent_CI.md](agent_CI.md) retains the mathematical specification and
historical application record needed by the other datasets.

The old simulation launchers and plot variants were retired after a verified
rollback archive. Historical `/cwork` runs retain their original frozen sources.
Exact historical reproduction uses the corresponding frozen source tree.
The active simulation folder retains only the agreed toy, model-selection,
matched-loss and convergence workflows.

## CelebA application

The finalized [expanded CelebA study](../../../docs/CelebA.md) uses 39,829 P
and 40,000 images per Q in one merged held-out test pool. The same observations
supply cell counts, neural means and Brier/Gap. Selected feature/pixel models
use JS with slopes 0.5/2. The [active workflow](../../CelebA/README.md) supplies
all 20 main assessments, 25 learned null controls and publication figures.

C.1/C.2 target population cell-average RDR, not individual-image ratios.
Repeated identities and historical inspection limit these to nominal,
retrospective image-level diagnostics. Smaller-data CelebA launchers and reports
are preserved in RDR-working and the external research archive.

## AGP real-versus-ICFM application

[AGP_ICFM_712_ci.py](../../AGP/AGP_ICFM_712_ci.py) provides the current approximately 7:1:2 rerun, retaining all eligible real data and the same full test/calibration observations. P therefore uses approximately 74/10/16 because the generator-unseen test pool is fixed; Q uses approximately 70/10/20. The [allocation helper](../../AGP/AGP_712_data.py) changes only training/validation allocation, preserving subject separation and the 609 generator-training exclusions. It uses the existing [source-data audit](../../AGP/AGP_ci_data.py), including a single generated bank to avoid shared random starts between historical Q banks.

| Role | Real P | ICFM Q |
|---|---:|---:|
| Training | 9,162 | 7,310 |
| Validation | 1,281 | 1,044 |
| Test = calibration | 2,002 | 2,089 |

The original MLP, initialization, optimizer, stopping rule, equally weighted midpoint Hellinger objective, and 20 score bins are retained; each empirical P/Q mean uses its own denominator. The refitted best-validation checkpoint receives unchanged C.1/C.2 intervals on the full test set. The report also compares empirical calibration across the three roles, explicitly treating training/validation curves as descriptive. Comparison to the old fit uses identical test samples, but the two learned score maps define different population cells.

Run `python3 -B experiments/AGP/AGP_ICFM_712_ci.py --p-policy keep-all`, with a fresh `--output-dir` for another run. See the [current AGP report](/cwork/yx306/RDR/JRSSB/CI/agp_icfm_712_20260920/report.md) and [reproduction instructions](../../AGP/README.md). Real subjects remain separate across roles, but within-subject dependence is unadjusted. Historical preprocessing, generator provenance, and prior inspection of the test cohort limit formal coverage claims. The prior full-test fit remains under `agp_icfm_full_test_20260918/`; the original fit and smaller-calibration results remain under `agp_icfm_20260918/`.
