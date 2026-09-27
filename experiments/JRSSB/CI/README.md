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

## CelebA feature-RDR application

[CELEBA_feature_ci.py](../CELEBA_feature_ci.py) reuses `calibration.py` unchanged for P versus Q_L and P versus Q_U, with fixed learned pool3 models, 20 equal-width bins, and nominal 95% C.1/C.2 intervals. The current full-test mode uses **18,000 P and 18,000 Q observations per contrast for both calibration and test**, with the same P observations in both contrasts. Training (60,000 per distribution) and validation (20,000 per distribution) remain disjoint from these test rows, including identities and generated seeds.

```bash
python3 -B experiments/JRSSB/CELEBA_feature_ci.py --calibration-mode full-test
```

The [application report](/cwork/yx306/RDR/JRSSB/CI/celeba_feature_full_test_20260918/report.md) includes all 40 cell intervals, sample-level lookups, split audits, hashes, source snapshots, and PNG/PDF figures. C.2 is simultaneous over the 20 cells of each contrast; additional joint endpoints use half the error budget per contrast for nominal 95% simultaneous coverage across both contrasts. The shared P sample does not invalidate this union-bound allocation. Simultaneous coverage of all fixed cell targets allows intervals to be attached to the calibration observations themselves; no extra query holdout is needed. This is not an independent assessment of calibration performance. C.1 remains asymptotic and marginal for a fixed cell.

These intervals target cell-average feature RDR, not individual true RDR or Hellinger divergence. True CelebA cell ratios are unknown, so this application does not measure coverage. C.2's finite-sample statement requires a fixed score map/partition independent of the calibration sample and independent image draws. Historical design use of 9,000 of the 18,000 test rows and repeated photos within identities remain recorded limitations; no correction for protocol selection or clustering is applied. The previous split-mode results (4,385 P / 4,500 Q calibration) remain preserved under `celeba_feature_20260918/`.

## CelebA pixel-RDR application

[CELEBA_pixel_ci.py](../CELEBA_pixel_ci.py) shares the feature-CI implementation and uses the identical full test/calibration pool and bins with `pixel_rdr` predictions and pixel checkpoint locks. The [pixel report](/cwork/yx306/RDR/JRSSB/CI/celeba_pixel_full_test_20260918/report.md) contains all 40 intervals, test mappings, and the same sampling limitations. Run `python3 -B experiments/JRSSB/CELEBA_pixel_ci.py --calibration-mode full-test`, using a fresh `--output-dir` for a rerun. Exact 0/2 scores and prescribed C.1 [0,2] fallbacks are retained. The updated plot marks unavailable C.1 cells without obscuring C.2, omits empty-cell placeholder estimates, and includes bin counts. Earlier split-mode results remain preserved under `celeba_pixel_20260918/`.

## AGP real-versus-ICFM application

[AGP_ICFM_712_ci.py](../../AGP/AGP_ICFM_712_ci.py) provides the current approximately 7:1:2 rerun, retaining all eligible real data and the same full test/calibration observations. P therefore uses approximately 74/10/16 because the generator-unseen test pool is fixed; Q uses approximately 70/10/20. The [allocation helper](../../AGP/AGP_712_data.py) changes only training/validation allocation, preserving subject separation and the 609 generator-training exclusions. It uses the existing [source-data audit](../../AGP/AGP_ci_data.py), including a single generated bank to avoid shared random starts between historical Q banks.

| Role | Real P | ICFM Q |
|---|---:|---:|
| Training | 9,162 | 7,310 |
| Validation | 1,281 | 1,044 |
| Test = calibration | 2,002 | 2,089 |

The original MLP, initialization, optimizer, stopping rule, equally weighted midpoint Hellinger objective, and 20 score bins are retained; each empirical P/Q mean uses its own denominator. The refitted best-validation checkpoint receives unchanged C.1/C.2 intervals on the full test set. The report also compares empirical calibration across the three roles, explicitly treating training/validation curves as descriptive. Comparison to the old fit uses identical test samples, but the two learned score maps define different population cells.

Run `python3 -B experiments/AGP/AGP_ICFM_712_ci.py --p-policy keep-all`, with a fresh `--output-dir` for another run. See the [current AGP report](/cwork/yx306/RDR/JRSSB/CI/agp_icfm_712_20260920/report.md) and [reproduction instructions](../../AGP/README.md). Real subjects remain separate across roles, but within-subject dependence is unadjusted. Historical preprocessing, generator provenance, and prior inspection of the test cohort limit formal coverage claims. The prior full-test fit remains under `agp_icfm_full_test_20260918/`; the original fit and smaller-calibration results remain under `agp_icfm_20260918/`.
