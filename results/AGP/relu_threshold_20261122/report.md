# Retrospective AGP comparison of relative density ratio estimators

The table reports held-out balanced Brier scores for the four separate real-versus-generated comparisons. Lower scores indicate more accurate real-source probabilities. Each estimator column gives the mean and sample SD of five individually fitted models using the archived final seeds. ΔBS subtracts the published ILR mean from the clipped ReLU mean.

| Generator | Published ILR neural | Clipped ReLU | ΔBS | Abundance/presence MLP | Winner including control |
|---|---:|---:|---:|---:|---|
| Mixture + ICFM | 0.135704 ± 0.004999 | 0.038329 ± 0.001856 | -0.097375 | 0.103411 ± 0.006773 | Clipped ReLU |
| NADE + ICFM | 0.212498 ± 0.002627 | 0.052609 ± 0.000759 | -0.159889 | 0.140496 ± 0.004063 | Clipped ReLU |
| MIDASim mask + ICFM | 0.141706 ± 0.003077 | 0.055049 ± 0.001903 | -0.086657 | 0.108084 ± 0.003079 | Clipped ReLU |
| Full MIDASim | 0.095237 ± 0.003138 | 0.094932 ± 0.001632 | -0.000305 | 0.133184 ± 0.002138 | Clipped ReLU |

The clipped feedforward ReLU estimator has a lower five-fit mean than the published ILR estimator for 4 of 4 generators and than the abundance/presence MLP for 4 of 4. The smallest absolute difference from ILR is for Full MIDASim (clipped ReLU minus ILR = -0.000305).

The clipped feedforward ReLU estimator uses abundance and taxon-specific positive-abundance hinges, an additive output, and two width-128 ReLU interaction layers. Each generator has its own fitted weights. The control uses the published deep sigmoid MLP on square-root abundance and presence features, with the published objective and slope for each generator. The proposed model also changes the architecture and output link, so a score difference cannot be attributed solely to thresholds.

The published assessment results motivated this architecture. These comparisons are retrospective. Variation across seeds measures initialization and training variability on fixed data. Theorem 3's distributional and optimization assumptions have not been established for AGP.

## Ensembles and clipping

For an ensemble, we first average the five real-source probabilities for each observation, then compute one balanced Brier score. Endpoint percentages give the fraction clipped to each endpoint within each source on the test set, averaged across the five clipped ReLU fits.

| Generator | Published ensemble BS | Clipped ReLU ensemble BS | Control ensemble BS | Real lower | Real upper | Generated lower | Generated upper |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mixture + ICFM | 0.105083 | 0.036520 | 0.081523 | 1.69% | 87.57% | 84.12% | 1.52% |
| NADE + ICFM | 0.207225 | 0.049279 | 0.128273 | 2.17% | 86.96% | 82.14% | 3.22% |
| MIDASim mask + ICFM | 0.129023 | 0.052030 | 0.095365 | 1.88% | 84.98% | 80.53% | 3.30% |
| Full MIDASim | 0.076643 | 0.088312 | 0.120353 | 3.28% | 60.45% | 63.85% | 3.08% |

For Full MIDASim, Published ILR neural has the lowest score under the ensemble comparison (0.076643).

[Per-seed results](per_seed.csv) · [Full summary](summary.csv)
