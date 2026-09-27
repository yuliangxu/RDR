# Agent 3: one held-out test split, learned estimators

**Reproduced 2026-09-13.**

The primary experiment has three roles: 60,000 training, 20,000 validation, and 18,000 test images per distribution. The test split is the complete union of the previous 9,000 design and 9,000 final-test rows, with no score-based filtering. The same real test images are used in both generator contrasts.

Training and validation remain unchanged. All four first-epoch validation losses are negative; the learned minimum and stopping trajectory are therefore identical with or without the constant-ratio baseline. This reproduction reuses the identical hashed learned checkpoints and their saved predictions. It applies no constant-ratio substitution and no stabilization-based null gate.

The single test split is disjoint from training and validation in image IDs, real identities, and generated seeds. Its historical source role is retained only for provenance. These data have previously been inspected: this is a retrospective held-out reanalysis, not a newly sealed confirmatory experiment.

| Role | Real P | Generated Q_L | Generated Q_U |
| --- | ---: | ---: | ---: |
| Training | 60,000 | 60,000 | 60,000 |
| Validation / early stopping | 20,000 | 20,000 | 20,000 |
| Test | 18,000 | 18,000 | 18,000 |

Both generator distributions use the same pinned Diffusion-StyleGAN2 CelebA64 checkpoint, with truncation psi 0.650 (lower) and 1.569 (upper), and constant synthesis noise. Features are the fixed 2,048-dimensional pytorch-fid 0.3.0 pool3 representation. The image preprocessing and learned checkpoints are unchanged.

## Estimator and uncertainty

For each branch, $M=(P+Q)/2$ and $r=dP/dM\in[0,2]$. The primary estimate is $\widehat H^2=1-\tfrac12\overline{r_P^{-1/2}}-\tfrac14\overline{\sqrt{r_P}}-\tfrac14\overline{\sqrt{r_Q}}$, with the original numerical clipping epsilon $10^{-6}$. Each mean uses all 18,000 observations of its source. Numerical clipping is distinct from replacing a learned model by $r=1$; it is retained to evaluate the original loss at saturated outputs.

Intervals use 5,000 paired image-level percentile bootstrap replicates, sharing real resamples across branches and image resamples across feature/pixel scores. They condition on the fitted models and do not include training variability or within-identity clustering. FID is recomputed from all 18,000 pool3 vectors per source; it is not the average of previous FIDs. FID is reported as a point estimate without a new equivalence test or bootstrap interval.

| Contrast | Test P / Q | FID | Feature H2 [95% interval] | Pixel H2 [95% interval] | Pixel minus feature [95% interval] |
| --- | ---: | ---: | ---: | ---: | ---: |
| Real vs lower | 18,000 / 18,000 | 27.083414 | 0.202260 [0.195235, 0.208579] | 0.291179 [0.290865, 0.291464] | 0.088919 [0.082609, 0.095930] |
| Real vs upper | 18,000 / 18,000 | 26.130102 | 0.160469 [0.152024, 0.167953] | 0.248409 [0.229564, 0.262545] | 0.087941 [0.068001, 0.104521] |

FID gap (lower minus upper): 0.953312. This is descriptive, not evidence of formal FID equivalence.

Lower-minus-upper feature H2: 0.041791 [0.031392, 0.052592]; pixel H2: 0.042770 [0.028690, 0.061584]; increment difference: 0.000979 [-0.017181, 0.022028].

The feature/pixel differences are diagnostics from separately fitted estimators, not exact conditional-divergence decompositions. Population information ordering does not guarantee ordering of finite fitted estimates.

## Learned-only null diagnostics

All 25 previously completed auxiliary null fits are recomputed from their learned scores in [learned_only_nulls.csv](../../../../../../cwork/yx306/RDR/JRSSB/ddim_diffusion_stylegan2_equal_fid/rdr/real_generator/single_test/learned_only_nulls.csv). They are separate same-source experiments, not another primary evaluation split. Generated nulls have 5,000 observations per held-out fold; real pixel nulls have 9,000 per fold. No exact-zero substitution or calibration-pass claim is applied. The real-pixel learned-only null discrepancy remains visible; these diagnostics do not establish calibrated neural divergence estimation.

| Auxiliary learned-only null | Repeats | Held-out P / Q per repeat | H2 range across repeats |
| --- | ---: | ---: | ---: |
| $Q_{L,Z}$ vs $Q_{L,Z}$ | 5 | 5,000 / 5,000 | [-0.000135, -0.000001] |
| $Q_{U,Z}$ vs $Q_{U,Z}$ | 5 | 5,000 / 5,000 | [-0.000162, -0.000042] |
| $Q_{L,X}$ vs $Q_{L,X}$ | 5 | 5,000 / 5,000 | [-0.000542, -0.000102] |
| $Q_{U,X}$ vs $Q_{U,X}$ | 5 | 5,000 / 5,000 | [-0.002240, -0.000750] |
| $P_X$ vs $P_X$ | 5 | 9,000 / 9,000 | [-0.002569, -0.001391] |

## Test figures

Random images are sampled uniformly; score-bin examples use up to 40 images nearest 0, 1, or 2 within the fixed score intervals. They are descriptive examples, not prevalence estimates.

![01_metric_hierarchy.png](../../../../../../cwork/yx306/RDR/JRSSB/ddim_diffusion_stylegan2_equal_fid/rdr/real_generator/single_test/01_metric_hierarchy.png)

![12_random_draws_P_QL_QU_40.png](../../../../../../cwork/yx306/RDR/JRSSB/ddim_diffusion_stylegan2_equal_fid/rdr/real_generator/single_test/12_random_draws_P_QL_QU_40.png)

![08_pixel_rdr_real_vs_upper.png](../../../../../../cwork/yx306/RDR/JRSSB/ddim_diffusion_stylegan2_equal_fid/rdr/real_generator/single_test/08_pixel_rdr_real_vs_upper.png)

![09_pixel_rdr_real_vs_lower.png](../../../../../../cwork/yx306/RDR/JRSSB/ddim_diffusion_stylegan2_equal_fid/rdr/real_generator/single_test/09_pixel_rdr_real_vs_lower.png)

![10_feature_rdr_real_vs_upper.png](../../../../../../cwork/yx306/RDR/JRSSB/ddim_diffusion_stylegan2_equal_fid/rdr/real_generator/single_test/10_feature_rdr_real_vs_upper.png)

![11_feature_rdr_real_vs_lower.png](../../../../../../cwork/yx306/RDR/JRSSB/ddim_diffusion_stylegan2_equal_fid/rdr/real_generator/single_test/11_feature_rdr_real_vs_lower.png)

![05_feature_pixel_score_map.png](../../../../../../cwork/yx306/RDR/JRSSB/ddim_diffusion_stylegan2_equal_fid/rdr/real_generator/single_test/05_feature_pixel_score_map.png)

![06_score_distributions.png](../../../../../../cwork/yx306/RDR/JRSSB/ddim_diffusion_stylegan2_equal_fid/rdr/real_generator/single_test/06_score_distributions.png)

## Reproduction

Run `OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 python3 experiments/JRSSB/CELEBA_single_test.py` from the repository root. The output directory contains test manifests, predictions, bootstrap replicates, exact thumbnail selections, checkpoint/split audits, and input/output hashes in `result.json`.

The [historical report source archive](../../../../../../cwork/yx306/RDR/JRSSB/ddim_diffusion_stylegan2_equal_fid/audit/2026-09-13/before_cleanup.zip) preserves the previous two-stage analysis and the full population derivations. Current artifacts and their hashes are listed in the [single-test result manifest](../../../../../../cwork/yx306/RDR/JRSSB/ddim_diffusion_stylegan2_equal_fid/rdr/real_generator/single_test/result.json).

For the complete preparation, generation, training, scoring and reporting workflow, see [README.md](README.md).
