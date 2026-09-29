# MNIST experiments

The retained experiments are **real versus VAE**, **real versus DCGAN**,
**controlled digit perturbation**, and the **real-versus-real validation null**.
All four now have fresh RDR fits using the DCGAN-selected **JS loss** and
**r(x)=2 sigmoid(2z(x))**. The configuration is transferred to the other
contrasts without further searches. The target is r=2p/(p+q), all CNN inputs
use [-1,1], and the pretrained generators remain fixed.

## Completed selected-model results

| Experiment | Assessment role | Best epoch | Brier | Local absolute calibration gap |
| --- | --- | ---: | ---: | ---: |
| Real versus VAE | Official test | 19 | 0.000748 | 0.001505 |
| Real versus DCGAN | Official test | 17 | 0.082477 | 0.018135 |
| Digit perturbation | Held-out official-test half | 16 | 0.193262 | 0.055599 |
| Real versus real | Validation only | 8 | 0.253278 | 0.091185 |

Balanced Brier is `0.5 mean_P[(r/2-1)^2] + 0.5 mean_Q[(r/2)^2]`.
It assesses classifier/RDR fitting risk, not generator visual quality. Local
gap compares calibration-estimated and neural cell means, weighted by evaluation
mixture mass. The generator and control diagnostics use different sampling
constructions, specified below.

- VAE/DCGAN: [PNG](../results/MNIST/selected_comparison/mnist_vae_dcgan_comparison.png),
  [PDF](../results/MNIST/selected_comparison/mnist_vae_dcgan_comparison.pdf),
  [report](../results/MNIST/selected_comparison/RESULTS.md),
  [verification](../results/MNIST/selected_comparison/verification.json).
- Perturbation/null: [PNG](../results/MNIST/selected_controls/mnist_selected_controls.png),
  [PDF](../results/MNIST/selected_controls/mnist_selected_controls.pdf),
  [calibration plot](../results/MNIST/selected_controls/mnist_controls_calibration.png),
  [report](../results/MNIST/selected_controls/RESULTS.md),
  [verification](../results/MNIST/selected_controls/verification.json).

The null remains a **validation diagnostic**, with no independent final-test
claim. Its fitted Brier is slightly worse than the constant-r=1 benchmark of
0.25; the learned score spread is retained rather than replaced by that baseline.

## Exact sample roles

| Experiment | Training P / Q | Early-stopping P / Q | Primary evaluation P / Q |
| --- | ---: | ---: | ---: |
| VAE or DCGAN | 55,000 / 55,000 fresh Q per epoch | 5,000 / 5,000 fixed Q | 10,000 / 10,000 |
| Digit perturbation | 60,000 / 60,000 fixed resampled rows | 5,000 / 5,000 resampled rows | 5,000 / 5,000 resampled rows |
| Null | 30,000 / 30,000 disjoint real halves | 5,000 / 5,000 disjoint real halves | None |

The generator split uses a private PyTorch seed-52 permutation of all 60,000
training images: first 5,000 validation, remaining 55,000 fitting. All 10,000
official test images are scored. Perturbation uses every training image as P;
seed 133 divides official test into disjoint validation/evaluation halves. Its
Q resampling seeds are 143/153/163. Null training halves use seed 123 and its
test-derived validation halves use seed 133. All ten primary control P/Q index
sequences exactly match historical saved evidence. All training rows are used
each epoch, including partial final batches.

Perturbation Q digit probabilities are
`[0,0,0.025,0.025,0.10,0.10,0.10,0.10,0.275,0.275]`.
Its 60,000/5,000/5,000 Q rows contain 28,734/2,414/2,346 unique images;
resampled rows are not additional independent source images. Nominal digit
references are `[2,2,1.6,1.6,1,1,1,1,0.533333,0.533333]`, assuming uniform P
and shared within-digit image distributions. The
[digit tables](../results/MNIST/selected_controls/perturbation/digits.json)
also use the actual held-out source-pool frequencies for an empirical reference.

## Calibration scope and regional diagnostics

All diagnostics use 20 fixed bins over [0,2]. C.1/C.2 intervals concern
cell-average true RDR, not individual-image RDR; neural cell means have
additional evaluation sampling uncertainty.

For VAE/DCGAN, disjoint 5,000-P/5,000-Q calibration and evaluation roles partition
the test pool; whole-test Brier uses all 10,000 P and 10,000 Q.

For perturbation/null, **after freezing the checkpoint**, four independent
streams draw 5,000 P and 5,000 Q calibration rows plus 5,000 P and 5,000 Q
diagnostic-evaluation rows. Perturbation uses its fixed 5,000-image held-out
pool, P uniform over images and Q controlled by digit. Null uses its pooled
10,000 test-derived validation images with P=Q uniform. Draws can repeat and
share image identities. These CIs describe **conditional finite-pool Monte
Carlo uncertainty**, excluding uncertainty from sampling the reference images.
They do not create an independent final-test result for the null.

| Contrast | Mass in [0.7,1.2) | Supported fraction of that mass | Local gap in supported middle cells |
| --- | ---: | ---: | ---: |
| VAE | 0.0005 | 0.4000 | 0.5283 |
| DCGAN | 0.0810 | 1.0000 | 0.0455 |
| Perturbation | 0.3846 | 1.0000 | 0.0650 |
| Null | 0.9351 | 1.0000 | 0.0829 |

VAE's tiny aggregate gap does not establish accuracy in its sparse middle
region. Overall supported mass is 0.9994 for VAE and 1.0 for the other contrasts,
up to rounding. Reports include all cells, widths and support. The official
test images were inspected previously, so assessments are retrospective;
generator pretraining membership has not been audited.

## DCGAN loss and sigmoid-slope selection

The [selection study](../results/MNIST/model_selection/RESULTS.md) compares
Hellinger, KL, chi-square and JS at slopes 0.5, 1, 2, 4: 16 configurations,
five paired repeats, 80 fits. Official-training roles contain 45,000 fitting
and 5,000 each for stopping, selection calibration and selection evaluation.
Official test is split 5,000/5,000 for calibration/evaluation after choice lock.

Checkpoints minimize Brier. The registered configuration rule forms a paired
one-standard-error Brier shortlist, then minimizes local gap, requiring 99%
supported mass in every repeat. JS/slope 2 was the only shortlisted candidate.
The SE describes variation conditional on fixed data; it is not independent-
dataset uncertainty. No MSE, AUC, FID or divergence score enters selection.

| Stage / configuration | Brier mean (SD) | Local gap mean (SD) |
| --- | ---: | ---: |
| Selection: JS, slope 2 | 0.098344 (0.001700) | 0.081758 (0.007296) |
| Selection: chi-square, slope 0.5, smallest gap | 0.104139 (0.001614) | 0.060505 (0.006814) |
| Selected configuration final assessment | 0.091056 (0.000968) | 0.066951 (0.007814) |

Chi-square/slope 0.5 has a smaller local gap but is outside the Brier shortlist.
The selected final middle region has 8.876% mass and gap 0.180317. These fits
use 45,000 fitting images; later generator refits use 55,000 and fresh seeds.

## Reproducible code and artifacts

The [workflow guide](../experiments/MNIST/README.md) documents public input
acquisition, CPU smoke runs, GPU/Slurm fitting, replay and tests. Active runners:

| Entrypoint | Purpose |
| --- | --- |
| [inputs.py](../experiments/MNIST/inputs.py) | Fetch or verify six public inputs against recorded hashes |
| [model_selection.py](../experiments/MNIST/model_selection.py) | Reproduce DCGAN configuration selection |
| [comparison.py](../experiments/MNIST/comparison.py) | Refit VAE/DCGAN with the selected configuration |
| [controls.py](../experiments/MNIST/controls.py) | Refit perturbation and the validation null |

Preparation verifies compact published evidence and recomputes the choice from
all 80 per-fit rows. Splits are reconstructed without private result folders.
Each run freezes inputs, source, configuration, seeds and indices. Replay uses
its frozen source, retaining strict source-hash checks as repository code changes.
The [reproduction verification](../results/MNIST/reproduction_verification.json)
records public-download hashes, clean-export tests and both completed CPU smoke runs.

Complete local runs, including checkpoints and raw score bundles:

- `/cwork/yx306/RDR/MNIST/dcgan_model_selection_20260927/` (jobs 56593056/56598383).
- `/cwork/yx306/RDR/MNIST/selected_comparison_20260928/` (job 56703307).
- `/cwork/yx306/RDR/MNIST/selected_controls_20260929/` (job 56707707).

Compact evidence is published for [selection](../results/MNIST/model_selection/README.md),
[comparison](../results/MNIST/selected_comparison/README.md), and
[controls](../results/MNIST/selected_controls/README.md). Full inputs/checkpoints
are generated by portable commands rather than committed.

## Historical evidence and cleanup

The previous Hellinger figures and tables remain in
[results/MNIST](../results/MNIST/README.md) and the
[historical report](../experiments/JRSSB/MNIST_jrssb.md). New fits also change
checkpoint selection and VAE's training schedule; differences cannot be
attributed solely to loss or sigmoid slope.

Twenty-one superseded MNIST source files, wrappers, notebooks and one-off
packaging tools were retired after replacement checks. Their hashes and the
verified full rollback archive are recorded in the
[cleanup receipt](../results/MNIST/cleanup_20260929.json). Raw data, ignored
historical scores, frozen packages and non-MNIST experiments were preserved.
The complete Hellinger reproduction package remains at
`/cwork/yx306/RDR/MNIST_jrssb_final/`; its code is historical evidence, not
another active repository workflow.
