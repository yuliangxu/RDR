# CelebA feature RDR: joint attribute regression

**Complete: two comparisons, all five selected feature-model repetitions; 39,829 real test images, 1,985 identities, 40 attributes.**

The response is log(2 − RDR), regressed jointly on an intercept and all 40 binary CelebA annotations (absent=0, present=1). This follows the old DDIM attribute regression, now using the merged real held-out test pool and frozen JS feature models. The predictors are semantic annotations, not the 2048 Inception coordinates. Generated images are not assigned guessed annotations.

## Numerical handling and uncertainty

Some float32 sigmoid scores rounded to exactly 2. Recover alpha*z from the unchanged checkpoint and calculate `log(2) - logaddexp(0, alpha*z)` in float64. No epsilon clipping or image deletion is used. Recomputed RDR scores must agree with the archived predictions within rtol=2e-5, atol=3e-6; exact discrepancies are recorded.

OLS point estimates match the original joint-regression specification. Per-fit standard errors/95% intervals are clustered by real identity (small-sample correction, t reference with G−1 degrees of freedom). BH q-values adjust across 40 attributes separately within each fit. These are exploratory conditional-on-network intervals, not simultaneous inference over repetitions or uncertainty in neural training. Repetitions share data; coefficient SD is training variability, not a confidence interval.

## Interpretation

For r=2p/(p+q), 2−r=2q/(p+q). A positive coefficient therefore associates attribute presence with higher relative Q support; a negative coefficient associates it with lower relative Q support, holding the other annotations fixed. The coefficient is a fitted difference in the log response, not a causal effect or a marginal attribute prevalence ratio. Correlated attributes and the restricted real-image support limit attribution. This analysis does not identify which latent Inception coordinates cause the shift, nor characterize generated-only regions absent from real data.

The joint annotations explain only about 3% (Q_l) and 6% (Q_u) of the log-response variation here. These rankings therefore identify the strongest measured associations, not a comprehensive explanation of the shift. The log transformation is sensitive to extreme fitted logits, especially for Q_l.

Rank by absolute mean coefficient over all five repetitions, following the old coefficient-magnitude convention. Because all predictors are binary, these compare fitted absent-to-present contrasts. Full tables also report prevalence-scaled coefficients and drop-one partial R²; these can rank attributes differently.

![Top attribute coefficients](attribute_coefficients.png)
[Publication PDF](attribute_coefficients.pdf)

## P versus Q_l

R² mean (SD): 0.030 (0.003).

| Attribute | Coefficient mean (SD) | Positive repeats / 5 | Mean partial R² |
| --- | --- | --- | --- |
| Blurry | -1.858 (0.573) | 0 | 0.0040 |
| Male | 1.150 (0.283) | 5 | 0.0027 |
| Gray_Hair | 0.815 (0.469) | 5 | 0.0005 |
| Brown_Hair | 0.804 (0.270) | 5 | 0.0022 |
| Bags_Under_Eyes | 0.714 (0.250) | 5 | 0.0016 |
| No_Beard | 0.681 (0.329) | 5 | 0.0006 |
| Wearing_Hat | -0.637 (0.518) | 0 | 0.0006 |
| 5_o_Clock_Shadow | 0.564 (0.275) | 5 | 0.0006 |
| Wearing_Necklace | 0.540 (0.271) | 5 | 0.0007 |
| Mouth_Slightly_Open | 0.536 (0.178) | 5 | 0.0014 |

## P versus Q_u

R² mean (SD): 0.057 (0.015).

| Attribute | Coefficient mean (SD) | Positive repeats / 5 | Mean partial R² |
| --- | --- | --- | --- |
| Wearing_Hat | 0.674 (0.387) | 5 | 0.0033 |
| Eyeglasses | 0.658 (0.205) | 5 | 0.0038 |
| Pale_Skin | 0.595 (0.095) | 5 | 0.0021 |
| 5_o_Clock_Shadow | -0.482 (0.026) | 0 | 0.0020 |
| Wearing_Necklace | -0.357 (0.070) | 0 | 0.0018 |
| Blond_Hair | 0.318 (0.098) | 5 | 0.0015 |
| Bald | 0.292 (0.088) | 5 | 0.0002 |
| Rosy_Cheeks | -0.281 (0.109) | 0 | 0.0007 |
| Goatee | 0.278 (0.042) | 5 | 0.0004 |
| Narrow_Eyes | 0.269 (0.059) | 5 | 0.0010 |

[All 80 attribute summaries](attribute_summary.csv) · [400 per-fit coefficients and clustered intervals](coefficients_per_fit.csv) · [Regression diagnostics](regression_metrics.csv)

The archived design.csv joins every test image to its original identity and annotations. responses/ retains the stable response and scaled logits for each fit. Frozen source, input hashes and completion records support replay. No RDR network is retrained and no selected-model metrics or previous results are changed. Historical test reuse makes this a retrospective analysis.
