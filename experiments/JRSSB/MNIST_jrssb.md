# MNIST: generator comparisons, controlled perturbation, and real-vs-real null

**Saved-result audit: 2026-09-13.** This report follows the structure of [agent3.md](agent3.md), using the existing pixel-space RDR results. The generator and perturbation experiments have separate RDR training, validation/early-stopping, and test roles. The available real-vs-real null has training and validation roles only; an independent-test null result is still missing. The summary retains the original saved results. The subsequent reproduction check below refitted all four RDR models without replacing those results.

## Target and split accounting

For every contrast, P denotes real MNIST and Q denotes either a frozen generator, label-resampled real MNIST, or a second real sample. The estimator targets

$$M=(P+Q)/2,\qquad r(x)=\frac{dP}{dM}(x)=\frac{2p(x)}{p(x)+q(x)}\in[0,2].$$

Scores above 1 indicate relative underrepresentation in Q, scores below 1 indicate overrepresentation in Q, and the same-population null has r=1. The denominator supplied to the Hellinger training objective is M. Consequently, the divergence target is H²(P,M), rather than H²(P,Q).

| Experiment | Training P / Q | Validation P / Q | Test P / Q |
| --- | ---: | ---: | ---: |
| Real vs VAE | 55,000 / online generation | 5,000 / online generation | 10,000 / 10,000 |
| Real vs DCGAN | 55,000 / online generation | 5,000 / online generation | 10,000 / 10,000 |
| Real vs digit-perturbed real | 60,000 / 60,000 resampled rows | 5,000 / 5,000 resampled rows | 5,000 / 5,000 resampled rows |
| Real vs real, saved null | 30,000 / 30,000 | 5,000 / 5,000 | **Not available** |

For both generators, global seed 42 (split seed 52) partitions the official 60,000-image training set into 55,000 training and 5,000 validation images. All 10,000 official test images are evaluated, and the same real assignments are used for both comparisons. The saved checkpoint indices confirm zero train/validation overlap and complete official-test coverage. Generated training/validation images are sampled online; they are not fixed 55,000/5,000-image Q datasets. The test draws use separate evaluation seeds.

For perturbation, all 60,000 official training images form training P. Seed 133 partitions the official test set into disjoint 5,000-image validation and test pools. Q is resampled with replacement exclusively within each role's P pool. Saved indices confirm zero validation/test overlap and no Q draw outside its role. The 60,000/5,000/5,000 Q rows contain **28,734/2,414/2,346 distinct source images**, respectively. P and Q can share images within a role; the roles themselves are disjoint.

These are nominal pool sizes. With batch size 512 and `drop_last=True`, a generator epoch consumes 54,784 numerator-P rows and 27,392 generated-Q draws in its midpoint batches. Perturbation consumes 59,904 numerator-P rows and 29,952 Q contributions per epoch; the null consumes 29,696 numerator-P rows and 14,848 Q contributions. Midpoint batches contain equal P and Q contributions. Validation visits all 5,000 numerator-P rows with up to ten batches, pairing them with 2,500 P and 2,500 Q midpoint contributions. Final scoring uses every listed evaluation row. Loader draws are not counts of unique images.

These split guarantees concern RDR fitting. The frozen generators' own training membership has not been audited here. The official test images have also been inspected in earlier experiments, so the report is a retrospective held-out analysis.

## Real vs VAE and real vs DCGAN

The primary comparison uses the paired run in `/cwork/yx306/RDR/mnist-generator-trainval-testall`. Both branches use pixel CNN ratio estimators, with model-space image inputs in [-1,1]. The frozen generator files are `mnist_vae/vae_epoch_25.pth` and `mnist_dcgan/netG_epoch_99.pth` under `/hpc/group/mastatlab/yx306/MNIST`. Validation selected VAE epoch 3 and DCGAN epoch 15, with saved losses -0.276662 and -0.097511. The VAE run allowed only three epochs; the training budgets were not matched.

**Input-scale verification.** Real training, validation, and test pixels use `2 * ToTensor(image) - 1`. VAE outputs are converted from [0,1] to [-1,1] before both midpoint training and test scoring; DCGAN training uses its native [-1,1] output and its test sampler explicitly returns [-1,1]. Conversion back to [0,1] is for image display only. The saved ratio networks have one-channel convolutional kernels and operate directly on 28×28 pixels, without an external feature extractor. The older checkpoints do not store an explicit evaluation-scale field. As a numerical cross-check, rescoring 256 evenly spaced real test positions on CPU in [-1,1] reproduced the saved scores with mean absolute differences 0.000123 (VAE) and 0.000138 (DCGAN), whereas [0,1] inputs gave differences 0.115424 and 0.362771. This supports the model-space evaluation path; it is not a bitwise replay of the historical run. All saved real test labels also match the official test data at the recorded indices.


| Contrast | Test P / Q | Mean RDR on P | Mean RDR on Q | Variational H² | Affinity plug-in H² |
| --- | ---: | ---: | ---: | ---: | ---: |
| Real vs VAE | 10,000 / 10,000 | 1.852093 | 0.020616 | 0.276175 | 0.300565 |
| Real vs DCGAN | 10,000 / 10,000 | 1.582470 | 0.449771 | 0.096444 | 0.123618 |

Both fitted estimates show a larger discrepancy for VAE than DCGAN. VAE-generated scores have median 0.000009, compared with 0.160887 for DCGAN; real-image medians are 1.952766 and 1.834286, respectively. This describes separation by these fitted pixel estimators, not a calibrated universal ranking of visual quality.

The variational column was recomputed directly from saved test scores using the same functional reported for CelebA:

$$\widehat H^2_{\mathrm{var}}=1-\tfrac12\overline{\hat r_P^{-1/2}}-\tfrac14\overline{\sqrt{\hat r_P}}-\tfrac14\overline{\sqrt{\hat r_Q}},$$

with ratios lower-clamped at 10^-6. The original MNIST figure and summary instead use

$$\widehat H^2_{\mathrm{aff}}=1-\tfrac12\left(\overline{\sqrt{\hat r_P}}+\overline{\sqrt{\hat r_Q}}\right),$$

with ratios lower-clamped at 10^-8 and the displayed divergence floored at zero. Both generator values are positive before that floor. These two functionals agree at the population ratio but can differ substantially for a fitted model. In particular, the VAE affinity estimate exceeds the population bound H²(P,(P+Q)/2) ≤ 1−1/√2 ≈ 0.292893; it must not be interpreted as a calibrated population divergence. No confidence intervals or training-repeat uncertainty are available for this paired run.

![Paired VAE/DCGAN test comparison](../../../../../../cwork/yx306/RDR/mnist-vae-dcgan-comparison/mnist_vae_dcgan_comparison.png)

**Figure: MNIST generator comparison.** Left: VAE; right: DCGAN. A/D: histograms of RDR on 10,000 real test images and 10,000 generated images per branch; titles report the variational H² defined above. C/F: real test images (top) and generated images (bottom), with 40 images per panel selected by smallest scores, proximity to 1, and largest scores; subtitles give the selected score ranges. B/E: VAE/DCGAN real-test RDR distributions stratified by digit, with quartile marks and a shared [0,2] vertical scale. Image mosaics are reused from the original saved paired figure; charts are redrawn from saved checkpoint scores. No images were regenerated and no models were refitted.

[PDF](../../../../../../cwork/yx306/RDR/mnist-vae-dcgan-comparison/mnist_vae_dcgan_comparison.pdf) · [Plotting script](../../../../../../cwork/yx306/RDR/mnist-vae-dcgan-comparison/make_mnist_vae_dcgan_comparison.py) · [Panel selection](../../../../../../cwork/yx306/RDR/mnist-vae-dcgan-comparison/panel_selection.csv) · [Provenance manifest](../../../../../../cwork/yx306/RDR/mnist-vae-dcgan-comparison/manifest.json)

The example panels select the smallest, closest-to-1, and largest fitted scores. They illustrate the score distribution and do not estimate digit prevalence. Later DCGAN-only input-scale experiments are excluded from this paired comparison: scoring [0,1] inputs through a ratio model trained on [-1,1] changes the evaluation protocol.

Sources: [test score summary](../../../../../../cwork/yx306/RDR/mnist-generator-trainval-testall/rdr_summary.csv), [test scores](../../../../../../cwork/yx306/RDR/mnist-generator-trainval-testall/rdr_scores.csv), [split manifest](../../../../../../cwork/yx306/RDR/mnist-generator-trainval-testall/split_manifest.json), and [experiment script](../MNIST_generator_strict_split.py).

## Controlled digit perturbation: RDR estimation accuracy

Q retains real images but changes their label probabilities: digits 0–1 are removed, digits 2–3 are downweighted, digits 4–7 retain probability 0.10 each, and digits 8–9 are upweighted to 0.275 each. The same probabilities apply to training, validation, and test resampling. Seed 123 and validation selection give epoch 4, with loss -0.023516.

If P and Q share the within-digit image distribution, with label probabilities π_d and w_d, the digit-level target is

$$r_d=\frac{2\pi_d}{\pi_d+w_d}.$$

For an image-only estimator, identifying this with r(x) also assumes that digit membership is determined by x. Without that assumption, the displayed formula is the label-space ratio. The saved theoretical table assumes π_d=0.10, although the actual P pools are not exactly balanced. Both the nominal reference and a test-pool-frequency reference are therefore shown below. The latter uses the full P pool frequency and the intended Q sampling probability, rather than realized Q draw counts.

| Digit | Q probability | Nominal target | Test-pool target | Test P n | Mean fitted RDR on test P | Mean fitted RDR on test Q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.000 | 2.0000 | 2.0000 | 528 | 1.7088 | — |
| 1 | 0.000 | 2.0000 | 2.0000 | 578 | 1.7857 | — |
| 2 | 0.025 | 1.6000 | 1.6057 | 509 | 1.5499 | 1.5573 |
| 3 | 0.025 | 1.6000 | 1.5981 | 497 | 1.4346 | 1.4425 |
| 4 | 0.100 | 1.0000 | 0.9806 | 481 | 0.9818 | 0.9787 |
| 5 | 0.100 | 1.0000 | 0.9407 | 444 | 0.9909 | 1.0060 |
| 6 | 0.100 | 1.0000 | 0.9919 | 492 | 1.1206 | 1.1156 |
| 7 | 0.100 | 1.0000 | 1.0291 | 530 | 1.1721 | 1.1739 |
| 8 | 0.275 | 0.5333 | 0.5151 | 477 | 0.6302 | 0.6349 |
| 9 | 0.275 | 0.5333 | 0.5046 | 464 | 0.6614 | 0.6623 |

The intended group ordering is recovered. Test-P means are **1.7490** for removed digits 0–1, **1.4929** for downweighted digits 2–3, **1.0707** for digits 4–7, and **0.6456** for upweighted digits 8–9. However, removed digits fall below the target 2 and upweighted digits exceed their targets. Digits 6–7 also show upward bias relative to their references.

Across all 5,000 test-P images, observation-level MAE/RMSE are **0.1755/0.2315** against the nominal targets and **0.1770/0.2327** against the test-pool targets. These are descriptive label-reference errors, combining within-digit variation and bias; they are not independently repeated training results. Resampled Q rows share source images, so they should not be treated as 5,000 additional independent images for uncertainty calculations.

As an additional calibration diagnostic, the raw affinity plug-in H² is **-0.000406** despite the deliberate distribution change. This illustrates why successful digit ordering alone does not establish accurate divergence estimation; the negative estimate is retained rather than silently replaced by zero.

Sources: [per-digit test summary](../results/MNIST_label_perturbation/test_digit_rdr_summary.csv), [label counts](../results/MNIST_label_perturbation/label_frequencies.csv), [score/split bundle](../results/MNIST_label_perturbation/rdr_scores.pt), and [experiment script](../MNIST_label_perturbation.py).

## Real-vs-real null: available validation diagnostic

The saved null randomly partitions the 60,000 official training images into disjoint 30,000-image P and Q halves. The official test set is partitioned into disjoint 5,000-image P and Q halves **used for validation and early stopping**. Saved indices confirm disjoint P/Q halves within each source. Seed 123 selected epoch 6, with validation loss -0.00004225. The scores come from the learned estimator; no constant-ratio substitution is applied.

| Evaluated sample | n | Mean RDR | SD | 5th–95th percentile | RMSE against r=1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training P | 30,000 | 1.005795 | 0.039044 | [0.940791, 1.068572] | 0.039471 |
| Training Q | 30,000 | 1.001358 | 0.039102 | [0.935594, 1.064396] | 0.039125 |
| Validation P | 5,000 | 1.003246 | 0.040263 | [0.937379, 1.068652] | 0.040390 |
| Validation Q | 5,000 | 1.003724 | 0.040282 | [0.934895, 1.069524] | 0.040450 |

The learned validation scores concentrate near the population null value 1. These percentiles describe score spread, not confidence intervals. Because the same validation images selected the model, this is **not an independent-test null calibration**.

To complete the requested three-role null experiment, a new fit must reserve validation images from the official training set before fitting and leave the official test halves for final evaluation only. One compatible allocation is 27,500/27,500 training P/Q, 2,500/2,500 validation P/Q, and 5,000/5,000 test P/Q. This allocation is a proposed follow-up, not a completed result. Relabeling the existing validation scores as test scores would not repair the issue.

Sources: [null summary](../results/MNIST_two_halves/rdr_summary.csv), [null score bundle](../results/MNIST_two_halves/rdr_scores.pt), and [experiment script](../MNIST_two_halves.py).

## Artifact provenance

Numbers were checked against the saved CSVs and tensor bundles; generator split indices, perturbation role membership, and null P/Q membership were inspected directly. The source files remain in their original locations. The comparison figure, plotting script, panel selections, and provenance manifest are saved under `/cwork/yx306/RDR/mnist-vae-dcgan-comparison`; no reproduced result files were saved in the repository's existing results folders.

SHA256 fingerprints of the audited inputs:

| Input | SHA256 |
| --- | --- |
| Paired generator `rdr_scores.csv` | `9df2e8cca494359a3718c8c607b9b6209b421372a4729dff6664a89c0e0cab34` |
| Perturbation `rdr_scores.pt` | `8617eb27982baaad902bb84f53a464cc8d80d729e8e71ca9f0afed24b61f8aed` |
| Null `rdr_scores.pt` | `5a407ede0a7416f9d91e647c35091cdda63bd34c48a200e34ebb95125c639da7` |


## Verified end-to-end reproduction

The complete RDR workflow was rerun twice on the pinned NVIDIA RTX A5000 / PyTorch 2.6.0 / CUDA 12.4 / cuDNN 90100 environment (jobs 55338090 and 55343216). Both runs reproduced **all 240,000 archived score values exactly**, with the same selected epochs. Both generator ratio state dictionaries also match the original checkpoints exactly. The regenerated original figure and final comparison PNG are pixel-identical to the saved figures, and every published table row is checked against its reconstruction.

The matching numerical profile enables cuDNN TF32 and leaves deterministic-algorithm forcing disabled, following the original defaults. A separate strict-arithmetic profile was repeatable but produced different scores, so it is not the final-result reproduction profile. Training and validation input ranges were checked throughout. The reconstructed perturbation/null weights are now saved; they reproduce all archived scores, although identity with the historically unsaved parameter tensors cannot be established. The null remains the same validation diagnostic described above.

The self-contained package is `/cwork/yx306/RDR/MNIST_jrssb_final`, with data, pretrained generators, source snapshots, original and reconstructed ratio weights, figures and audits. A fresh ZIP extraction passed replay with an empty `PYTHONPATH`. The [workflow](mnist_final_workflow.py) provides saved-result replay and full retraining followed by report reconstruction. See the [reproduction guide](../../../../../../cwork/yx306/RDR/MNIST_jrssb_final/README.md) and [execution audit](../../../../../../cwork/yx306/RDR/MNIST_jrssb_final/end_to_end_audit.json).

After these checks passed, 55 superseded MNIST files (29,841,942 bytes) were retired from their original locations into a verified rollback ZIP. The final inputs, raw MNIST data, shared utilities, non-MNIST experiments, and README-referenced legacy entrypoints were preserved. The exact file list is in the [cleanup receipt](../../../../../../cwork/yx306/RDR/MNIST_jrssb_final/cleanup_receipt.json).
