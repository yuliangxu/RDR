# Expanded-data loss and activation selection

The expanded CelebA data are the main study. This stage selects loss and
bounded-sigmoid slope for each P/Q pair and representation, with fixed
architectures. The historical 60k study is preserved as pilot evidence.

## Status checked 2026-10-03

The [results report](../../results/CelebA/model_selection_expanded_20260929/RESULTS.md)
has been refreshed with 319/320 verified completed fits: all 160 feature fits
and 159/160 pixel fits. Task `56842200_211` (P versus Q_L, pixel Hellinger,
slope 4, repeat 03) failed with `Nonfinite objective at epoch 7`.
Selector `56842201` was blocked by the failed dependency. The automatic
selection did not finish, but the user accepted JS/slope 0.5 for feature and
JS/slope 2 for pixel on 2026-10-03; their final assessments and the five-family
null rerun are complete. Candidate means
require all five repeats, so the failed candidate is not summarized using
only its four successful repeats. Completed outputs are preserved; this
status refresh does not retry training or change the registered protocol.

## Registered configuration

The user accepted JS loss with feature slope 0.5 and pixel slope 2 for both
pairs on 2026-10-03. The
[separate final assessment](../../results/CelebA/merged_test_20261003/RESULTS.md)
completed all 20 evaluations (five saved repetitions per comparison) on the
final splits, without refitting. See [the main summary](../../docs/CelebA.md).
This explicit acceptance is recorded in that run's freeze; it does not
retroactively mark this 319/320 selection grid complete.

Configuration: [expanded_selection_config.json](expanded_selection_config.json).
There are 2 pairs x 2 representations x 4 losses x 4 slopes x 5 repetitions
= 320 fits. Losses are Hellinger, KL, chi-square and JS; slopes are 0.5, 1,
2 and 4 in `r = 2 sigmoid(alpha z)`. Feature width512 and pixel width64
retain the baseline architectures. This run stops after loss/slope selection:
no architecture sensitivity or final calibration/evaluation is scheduled.

| Role | P images | Each Q branch |
| --- | ---: | ---: |
| Training | 122,984 | 120,000 |
| Early stopping | 19,898 | 20,000 |
| Selection calibration | 9,935 | 10,000 |
| Selection evaluation | 9,953 | 10,000 |
| Reserved final calibration | 19,906 | 20,000 |
| Reserved final evaluation | 19,923 | 20,000 |

Real identities and generated seeds are disjoint across roles. Both pairs
share exactly the same P rows. The frozen split seed is2026092902. Final
manifests are checked for overlap, but final image/feature arrays are not
loaded by this selection stage. Previously inspected data make this a
retrospective study; five repetitions measure training variability
conditional on the fixed data.

Checkpoints minimize stopping-set balanced Brier. Candidate selection uses
the paired one-SE Brier shortlist and then minimum local absolute Gap,
requiring at least0.99 supported mass in every repetition. Calibration rows
estimate cell ratios; independent selection-evaluation rows provide neural
cell means and mixture masses. Both sources retain equal weight despite
unequal image counts. The20 score bins on[0,2] remain fixed.

## Numerical treatment

The pilot had15 Hellinger pixel failures at the float32 gradient-norm check.
For this new study, Hellinger backward factors out the largest inverse-root
exponential, then accounts for that factor when clipping the parameter
gradient. Norm accumulation uses float64 for all losses. This preserves the
mathematical objective and clipped gradient without adding ratio clipping
or dropping observations. Nonfinite objective/gradient components remain
errors. Tests compare against a float64 reference, including logits-1000
and float32 norms that would overflow.

## Preparation and execution

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python3 -B experiments/CelebA/model_selection.py prepare \
  --output /cwork/yx306/RDR/CelebA/model_selection_expanded_20260929 \
  --config experiments/CelebA/expanded_selection_config.json
```

After preparation, use the saved entrypoint under `source/experiments/CelebA/`.
Prepared arrays contain raw pool3 features or uint8 pixels. Source manifests,
feature and generated-shard hashes, raw JPEG content-hash ledgers and split
counts are retained. Feature standardization is fitted on training only.
`submit` schedules the baseline grid and a completion-gated CPU selector.
`RESULTS.md`/CSV contain the comparison; `baseline_selection.json` seals the
selected settings and fit hashes; `COMPLETE.json` marks completion of this
selection stage only. It does not claim final-test assessment is complete.
