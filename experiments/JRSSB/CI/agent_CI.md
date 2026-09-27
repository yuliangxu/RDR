> Shared CI specification and CelebA/AGP application record.
> The retained simulation workflow and results are summarized in
> [docs/simulations.md](../../../docs/simulations.md).



# RDR calibration: Algorithms C.1 and C.2, with adjacent-cell merging

## Current experiment reports

| Experiment | Markdown report |
|---|---|
| CelebA feature-level: $P$ versus $Q_U$ and $Q_L$ | [/cwork/yx306/RDR/JRSSB/CI/celeba_feature_full_test_20260918/report.md](/cwork/yx306/RDR/JRSSB/CI/celeba_feature_full_test_20260918/report.md) |
| CelebA pixel-level: $P$ versus $Q_U$ and $Q_L$ | [/cwork/yx306/RDR/JRSSB/CI/celeba_pixel_full_test_20260918/report.md](/cwork/yx306/RDR/JRSSB/CI/celeba_pixel_full_test_20260918/report.md) |
| AGP real versus ICFM, merged with $h=20$ | [/cwork/yx306/RDR/JRSSB/CI/agp_icfm_712_merged_h20_20260920/report.md](/cwork/yx306/RDR/JRSSB/CI/agp_icfm_712_merged_h20_20260920/report.md) |
| AGP ILR architecture comparison, validation RDR and CIs | [/cwork/yx306/RDR/JRSSB/CI/agp_architecture_20260925/report.md](/cwork/yx306/RDR/JRSSB/CI/agp_architecture_20260925/report.md) |
| AGP residual MLP: equal-width and adaptive $h=20$ CIs, with variance upper bounds | [/cwork/yx306/RDR/JRSSB/CI/agp_residual_equal_width_ci_20260925/report.md](/cwork/yx306/RDR/JRSSB/CI/agp_residual_equal_width_ci_20260925/report.md) |

The CelebA reports use full-test calibration with the original 20 fixed bins. The original AGP real-versus-ICFM $h=20$ report uses eight merged regions, with C.1 joint-bootstrap intervals and C.2 adjustment over all 210 contiguous candidates. The residual-MLP report uses a different frozen network: its $h=20$ partition contains two regions. It also reports true within-cell RDR variance upper confidence bounds derived from the mean intervals for both its 20 fixed bins and its selected adaptive regions.

AGP threshold comparison: the [$h=40$ report](/cwork/yx306/RDR/JRSSB/CI/agp_icfm_712_merged_h40_20260922/report.md) uses the same frozen model and test/calibration observations, producing five merged regions. The $h=20$ report above remains available for comparison; both settings protect the same 210 candidates.

AGP model diagnostics: the [compositional-input comparison report](/cwork/yx306/RDR/JRSSB/CI/agp_composition_20260922/report.md) compares raw and closed abundances, ILR, learned log-contrasts, and phylogenetic ILR with the same loss and splits. It includes three fitting seeds, validation-only zero-replacement sensitivity, and score-versus-frequency diagnostics. These diagnostics use 20 fixed score bins; they do not replace the merged-cell CI reports above.

AGP architecture diagnostics: the [2026-09-25 report](/cwork/yx306/RDR/JRSSB/CI/agp_architecture_20260925/report.md) compares narrow, wider, deep, and residual ILR MLPs across three seeds, with a separate stopping-patience control. Validation determines checkpoints and model selection before test scoring. All 15 $h=40$ partitions collapse to the whole score range, whose structural interval $[1,1]$ provides no regional calibration evidence. The report therefore also retains the original 20-bin intervals from the same protected 210-candidate family, alongside validation RDR, middle-region mass and gaps, and Hellinger-loss diagnostics. CIs remain retrospective nominal cell-average intervals, simultaneous within each fitted model only.

Implement these two post-processing procedures using the existing neural-network RDR predictor. Keep the trained network fixed; neither procedure requires changing its architecture, training loss, or weights.

The original fixed-partition notation and guarantees below follow Section 1.4 of `RDR_calibration_inference_algorithms.tex`. The variance-bound transformation added on 2026-09-25 and the adjacent-cell merging extension specified on 2026-09-20 are described before the original algorithms.

## Upper confidence bounds for true within-cell RDR variance

Keep the network frozen and use the same positive-mixture-mass cells as the mean calibration. For $M=(P+Q)/2$, $r_0(x)=2p(x)/(p(x)+q(x))$, and a cell $A$, the additional population target is

$$
v_A=\operatorname{Var}_{X\sim M}\{r_0(X)\mid X\in A\}
=\mathbb E_M\{r_0(X)^2\mid A\}-\theta_A^2,
\qquad
\theta_A=\mathbb E_M\{r_0(X)\mid A\}.
$$

This is heterogeneity of the **true RDR within the cell**, not the sampling variance of the estimated cell mean and not the variance of neural predictions. Cells are defined using the neural score, but the only assumed range for their true RDR is $[0,2]$; a narrow neural-score bin does not imply a correspondingly narrow true-RDR range. Conditional moments are undefined for population cells with $M(A)=0$; coverage statements concern positive-mass cells. Zero observed counts do not establish zero population mass.

Since $0\leq r_0\leq2$, pointwise $r_0^2\leq2r_0$. Conditional expectation therefore gives

$$
0\leq v_A\leq 2\theta_A-\theta_A^2=\theta_A(2-\theta_A)\leq1.
$$

Suppose an existing calibration procedure gives a mean interval $[L_A,U_A]\subseteq[0,2]$. Maximize this concave quadratic over that interval:

$$
c_A=\min\{U_A,\max\{L_A,1\}\},\qquad
V_A^+=c_A(2-c_A)
=\begin{cases}
U_A(2-U_A),&U_A<1,\\
1,&L_A\leq1\leq U_A,\\
L_A(2-L_A),&L_A>1.
\end{cases}
$$

Report $[0,V_A^+]$ as the variance confidence set; $\sqrt{V_A^+}$ is the corresponding upper confidence bound for the conditional standard deviation. No variance point estimate is identified by this transformation. In particular, substituting the empirical mean into $\hat\theta_A(2-\hat\theta_A)$ without accounting for its uncertainty does not yield this confidence bound.

### Coverage inheritance and exceptional cases

The implication

$$
\{\theta_A\in[L_A,U_A]\}
\ \subseteq\ \{0\leq v_A\leq V_A^+\}
$$

holds deterministically. It also holds after intersecting over a fixed candidate family. The variance bounds and associated mean intervals are thus valid together on the **same coverage event**, with no additional allocation of $\alpha_{\mathrm{CI}}$:

- Original fixed-bin C.1 intervals give asymptotic **marginal** variance bounds, subject to the same regularity assumptions. They do not become simultaneous after transformation.
- Original fixed-bin C.2 intervals give simultaneous variance bounds over the original $K$ cells, with the same finite-sample validity assumptions and the existing $\alpha_{\mathrm{CI}}/(4K)$ probability-tail adjustment.
- Adaptive C.1 uses its existing joint multiplier-bootstrap event over the full candidate family; transformed bounds inherit its **asymptotic simultaneous** interpretation and approximation limitations.
- Adaptive C.2 retains all $J=K(K+1)/2$ candidates in its original adjustment. For $K=20$, keep $J=210$, even when only two regions are selected. Transforming their protected intervals gives simultaneous bounds for the calibration-selected positive-mass regions under the same sampling assumptions.

No additional correction is required for transforming an associated family of mean intervals. Separate fixed-bin versus adaptive analyses, separate methods, or separate fitted models do not automatically share one joint 95% guarantee. Retrospective AGP limitations, including repeated observations within subjects and historical preprocessing/provenance, remain unchanged; report the inherited confidence levels as nominal rather than measured AGP coverage.

An unavailable-mean fallback $[0,2]$ gives $V_A^+=1$, the trivial global variance bound. Keep the original unavailability flags. An exact mean interval $[1,1]$ also gives $V_A^+=1$, **not zero**: even the structurally known whole-space mean $\mathbb E_M r_0=1$ is compatible with maximal variance when P and Q have disjoint support. Exact mean intervals $[0,0]$ or $[2,2]$ give variance bound zero. These rules apply unchanged to the whole-space candidate; do not reuse its zero mean standard error as the true-RDR standard deviation.

### What the bound establishes

Given only a cell mean $\theta$, both extremes are attainable: constant true RDR $\theta$ has variance zero, whereas true RDR equal to $2$ with conditional mixture probability $\theta/2$ and to $0$ otherwise has variance $\theta(2-\theta)$. Such within-cell distributions can have the same P/Q cell probabilities. Thus the range-and-mean bound is sharp without additional within-cell information, and increasing calibration size alone generally does not identify the actual variance. A bound near one is uninformative about heterogeneity; a small upper bound limits aggregate within-cell variation but does not rule out rare large deviations.

This is neither a confidence interval nor a prediction interval for each individual $r_0(x)$. It does not justify a normal range $\theta_A\pm1.96\sqrt{V_A^+}$. Individual-value probability ranges would require further quantile inference or an explicitly derived moment inequality.

### Implementation and AGP outputs

Use `variance_upper_bound(lower, upper)` in [utils/calibration.py](../../../utils/calibration.py). It accepts broadcastable scalar/array endpoints, validates finite $0\leq L\leq U\leq2$, and returns $c(2-c)$. It leaves the existing C.1/C.2 algorithms and result schemas unchanged. [test_variance_bound.py](../../../tests/test_variance_bound.py) checks boundaries, broadcasting, extremal distributions, the whole-space exception, and exact enumerated C.2 coverage inheritance.

The [residual-MLP report](/cwork/yx306/RDR/JRSSB/CI/agp_residual_equal_width_ci_20260925/report.md) adds derived tables and plots for both partitions under `variance_bounds/`, retaining the original mean intervals, counts, score attachments, frozen network, and adaptive candidate adjustment. The [equal-width variance plot](/cwork/yx306/RDR/JRSSB/CI/agp_residual_equal_width_ci_20260925/variance_bounds/equal_width/variance_upper_bounds.png) and [adaptive $h=20$ variance plot](/cwork/yx306/RDR/JRSSB/CI/agp_residual_equal_width_ci_20260925/variance_bounds/adaptive_h20/variance_upper_bounds.png) display the corresponding C.1 and C.2 upper bounds. [AGP_residual_variance_ci.py](../../AGP/AGP_residual_variance_ci.py) generates the additions from the saved intervals without retraining or redefining the partitions.

```bash
python3 -B -m unittest discover -s tests -p 'test_variance_bound.py'
```

## Shared adjacent-cell merging extension

**Purpose:** accumulate enough observations from both distributions in each reported region, while accounting for the fact that calibration counts determine the merged boundaries. Keep the fitted network and the elementary score-bin boundaries fixed before observing calibration data. C.1 uses a joint multiplier-bootstrap critical value over a fixed candidate family; C.2 uses exact probability intervals with a multiplicity adjustment over that same family. Both then report the regions selected by the same merging rule.

### Fixed elementary bins and candidate family

Let $I_1,\ldots,I_K$ partition $[0,2]$, and let $A_j=\{z:\hat r_{\mathrm{tr}}(z)\in I_j\}$. Use left-closed, right-open score intervals, with the final interval including $2$. The complete candidate family contains every contiguous union, including the whole score range:

$$
\mathcal A=\left\{A_{a:b}=\bigcup_{j=a}^{b}A_j:1\leq a\leq b\leq K\right\},
\qquad J=|\mathcal A|=K(K+1)/2.
$$

For the existing 20 equal-width elementary bins, $J=210$. This family is fixed conditional on training even though the regions ultimately reported are selected using calibration observations. Do not reduce the candidate family to the selected regions when computing either method's critical value or multiplicity adjustment.

Use independent calibration samples $X_1,\ldots,X_n\sim P$ and $Y_1,\ldots,Y_m\sim Q$, independent of network training and model selection. For each candidate $A$, aggregate the elementary counts and set

$$
k_P(A)=\sum_{i=1}^{n}\mathbf1\{X_i\in A\},\qquad
k_Q(A)=\sum_{\ell=1}^{m}\mathbf1\{Y_\ell\in A\},\qquad
\hat p_A=k_P(A)/n,\qquad \hat q_A=k_Q(A)/m.
$$

The estimate is $\hat\theta_A=2\hat p_A/(\hat p_A+\hat q_A)$ when the denominator is positive. Always retain the separate P and Q denominators. For an empirically empty candidate, store estimate $1$ as a flagged placeholder and use the conservative interval described below; this placeholder is not evidence that the population ratio equals $1$.

### Shared selection rule

Specify a positive integer threshold $h$ before running the procedure. Counts refer to the calibration samples, not to training, validation, or a separate query set.

1. Scan the elementary bins from left to right.
2. Accumulate adjacent bins until both accumulated counts satisfy $k_P(A)\geq h$ and $k_Q(A)\geq h$.
3. Close that merged region, start a new accumulation at the next elementary bin, and continue.
4. Merge any leftover right-hand tail into the preceding completed region. If no region reached both thresholds, return the whole score range as one region.

The selected regions form a partition of the complete score range. Empty elementary bins remain represented through their containing merged region. Every selected region meets both count thresholds unless no region qualifies, in which case the whole-range fallback may have fewer than $h$ observations from one distribution. Do not drop that fallback or leave any scores without a region.

The user selected $h=20$ per distribution for the AGP rerun. Its numerical settings are $B=10{,}000$ bootstrap repetitions, bootstrap seed $20260920$, and $\alpha_{\mathrm{CI}}=0.05$. These are recorded configuration choices, not parts of a coverage theorem; the threshold remains configurable. This method was documented before implementation and the rerun; the completed AGP application is recorded below.

### C.1 extension: joint multiplier-bootstrap intervals

For every $A\in\mathcal A$ with $\hat p_A+\hat q_A>0$, calculate the two estimated derivatives

$$
\hat a_A=\frac{2\hat q_A}{(\hat p_A+\hat q_A)^2},\qquad
\hat b_A=-\frac{2\hat p_A}{(\hat p_A+\hat q_A)^2},
$$

and apply the original C.1 variance formula to the aggregated region:

$$
\hat s_A^2=
\hat a_A^2\frac{\hat p_A(1-\hat p_A)}{n}
+\hat b_A^2\frac{\hat q_A(1-\hat q_A)}{m}.
$$

For each repetition $t=1,\ldots,B$, draw independent standard normal multipliers $\xi_1^{(t)},\ldots,\xi_n^{(t)}$ and $\zeta_1^{(t)},\ldots,\zeta_m^{(t)}$, independently between P and Q. Calculate

$$
G_A^{(t)}=
\frac{\hat a_A}{n}\sum_{i=1}^{n}\xi_i^{(t)}
\{\mathbf1(X_i\in A)-\hat p_A\}
+\frac{\hat b_A}{m}\sum_{\ell=1}^{m}\zeta_\ell^{(t)}
\{\mathbf1(Y_\ell\in A)-\hat q_A\}.
$$

**Use the same multipliers across all candidate regions within each repetition.** This preserves the dependence between overlapping candidates. The network stays fixed. A count-based implementation may draw one independent Gaussian sum for each elementary P bin with variance equal to its count, and independently do the same for Q. Summing those elementary Gaussian sums over a candidate and subtracting its estimated probability times the all-bin sum has exactly the conditional Gaussian law of the displayed observation-level multiplier sum. Drawing independent fluctuations for each candidate is incorrect.

Studentize only candidates with defined estimates and $\hat s_A>0$:

$$
T^{(t)}=\max_{A\in\mathcal A:\,\hat s_A>0}
\frac{|G_A^{(t)}|}{\hat s_A}.
$$

Let $\hat c_{1-\alpha_{\mathrm{CI}}}$ be the empirical $1-\alpha_{\mathrm{CI}}$ quantile of these $B$ maxima. Record the quantile convention and random seed; use NumPy's `quantile(..., method="higher")`, which selects the observation with one-based rank $1+\lceil(B-1)(1-\alpha_{\mathrm{CI}})\rceil$, for reproducibility. Return the candidate interval

$$
\mathcal I_A=
\left[\hat\theta_A-\hat c_{1-\alpha_{\mathrm{CI}}}\hat s_A,
\hat\theta_A+\hat c_{1-\alpha_{\mathrm{CI}}}\hat s_A\right]\cap[0,2].
$$

For a candidate with an undefined estimate or zero estimated standard error, return $[0,2]$ and omit it from studentization. If no candidate can be studentized, record that status and use these fallbacks without attempting a maximum over an empty set. The whole-space candidate is a structural exception: its population ratio equals $1$, so its estimate and interval are exactly $1$ and $[1,1]$. An empirically zero-variance proper subset is not the whole-space candidate merely because it happens to contain all observed samples.

Finally apply the shared selection rule and report the already-computed intervals for the selected candidates. This is an **asymptotic simultaneous extension** of C.1, rather than the original marginal normal interval. Its justification requires a valid uniform delta-method and multiplier approximation over the candidate family, with suitable sample sizes and nondegeneracy for the studentized candidates. A fixed observed count threshold such as $h=20$ does not itself establish those conditions or a finite-sample coverage guarantee. Finite $B$ also introduces Monte Carlo error in the critical value.

### C.2 extension: exact simultaneous candidate intervals

Allocate the error across the original candidate family:

$$
\delta=\frac{\alpha_{\mathrm{CI}}}{2J},\qquad
\frac\delta2=\frac{\alpha_{\mathrm{CI}}}{4J}.
$$

Retain this adjustment even if merging selects only a few regions. For every $A\in\mathcal A$, compute the original Clopper–Pearson probability intervals with failure probability $\delta$, using $(k_P(A),n)$ for $[L_P(A),U_P(A)]$ and $(k_Q(A),m)$ for $[L_Q(A),U_Q(A)]$. In particular, use

$$
L(k,N)=\begin{cases}0,&k=0,\\
q_{\mathrm{Beta}}(\delta/2;k,N-k+1),&k>0,
\end{cases}\qquad
U(k,N)=\begin{cases}1,&k=N,\\
q_{\mathrm{Beta}}(1-\delta/2;k+1,N-k),&k<N.
\end{cases}
$$

Transform them into

$$
\mathcal B_A=
\left[
\frac{2L_P(A)}{L_P(A)+U_Q(A)},
\frac{2U_P(A)}{U_P(A)+L_Q(A)}
\right].
$$

This includes candidates with zero observed counts. Use the structural whole-space target to return $[1,1]$ for that candidate, while still retaining $J$ in the adjustment. Apply the same selection rule as C.1 and report the intervals belonging to the selected candidates.

Conditional on the frozen network and fixed elementary boundaries, the union bound over the $2J$ probability intervals gives simultaneous coverage at least $1-\alpha_{\mathrm{CI}}$ for all positive-mass candidate ratios. Overlap between candidates does not invalidate this bound. Therefore the event also covers every positive-mass region selected by the calibration-dependent merging rule. This is a finite-sample statement under the stated independent IID sampling assumptions; it does not follow from applying the old $K$-bin correction or correcting only for the number of selected regions.

### Selected-region target, outputs, and implementation

For a selected region $\widehat A$ of positive population mixture mass, the target is

$$
\theta_{\widehat A}=
\frac{2P(\widehat A)}{P(\widehat A)+Q(\widehat A)}
=\mathbb E_{Z\sim M}[r_0(Z)\mid Z\in\widehat A],
\qquad M=(P+Q)/2,
$$

where the expectation treats the realized network and selected region as fixed and averages over a fresh population observation. The region, and thus the target assigned to a query, varies with calibration data. For query $x$, find its elementary bin, map it to the selected merged region, and return that region's estimate and interval. The population counterpart of the plotted mean network score is $\mathbb E_M[\hat r_{\mathrm{tr}}(Z)\mid Z\in\widehat A]$; it is distinct from $\theta_{\widehat A}$ unless the average prediction error is zero there.

Merging improves observed cell counts at the cost of score resolution. It changes the cell-average target and can combine observations with very different individual true RDR values. In particular, a whole-range result of $[1,1]$ is exact for the global mixture-average RDR and does not assert that $P=Q$ or that individual RDRs equal $1$. Neither method imposes monotonicity on the fitted calibration curve.

The implementation entry point is `calibrate_merged_counts` in [utils/calibration.py](../../../utils/calibration.py), alongside the preserved original `calibrate_counts`. Record elementary edges and counts, all $J$ candidate regions and their counts/estimates/intervals, the selected regions, elementary-to-merged lookup, $h$, $\alpha_{\mathrm{CI}}$, bootstrap repetitions/seed/quantile convention/critical value, and exceptional-case flags. Both algorithms must use the same selected partition. The AGP rerun reuses the frozen 7:1:2-policy checkpoint and its existing test/calibration P/Q samples so that only the calibration procedure changes; training and validation comparisons remain descriptive.

### Completed AGP application, 2026-09-20

The [merged AGP report](/cwork/yx306/RDR/JRSSB/CI/agp_icfm_712_merged_h20_20260920/report.md) reuses the frozen epoch-505 model from the approximately 7:1:2-policy fit. Its 2,002 P and 2,089 Q test observations are also the calibration samples. The checkpoint, scores, split manifest, training history, and fit protocol are preserved byte for byte. Training remains 9,162 P / 7,310 Q; validation remains 1,281 P / 1,044 Q. No network or generator was retrained.

The 20 elementary bins produce the following eight regions under $h=20$:

| Score region | P count | Q count | Calibrated estimate |
|---|---:|---:|---:|
| $[0.0,0.2)$ | 24 | 1,763 | 0.028012 |
| $[0.2,0.4)$ | 39 | 158 | 0.409621 |
| $[0.4,0.5)$ | 23 | 30 | 0.888879 |
| $[0.5,0.6)$ | 28 | 33 | 0.939193 |
| $[0.6,0.8)$ | 49 | 21 | 1.417713 |
| $[0.8,1.1)$ | 99 | 22 | 1.648849 |
| $[1.1,1.5)$ | 125 | 22 | 1.711347 |
| $[1.5,2.0]$ | 1,615 | 40 | 1.953628 |

All selected regions meet both thresholds. All $J=210$ contiguous candidates remain in C.2's adjustment, giving tail probability $\alpha_{\mathrm{CI}}/(4J)=0.00005952380952$. The C.1 bootstrap uses all 209 positive-SE candidates and gives critical value $3.40276779990528$ from 10,000 repetitions. No selected interval needs the exceptional-case fallback.

| Method | Original 20-bin mean width | Merged 8-region mean width |
|---|---:|---:|
| C.1 | 0.103616 | 0.120126 |
| C.2 | 0.226987 | 0.168486 |

Both columns use the same test observations and equal-mixture weights. The population cell targets change after merging, and C.1 changes from marginal to simultaneous inference, so these are descriptive width comparisons. The merged point estimates happen to increase in this realization; no monotonicity constraint was imposed. The original network remains descriptively miscalibrated: 7/8 training regions, 6/8 validation regions, and 7/8 test regions have empirical cell RDR above the within-region mean network score. The common regions in these diagnostics are selected from test/calibration counts.

[Interval plot](/cwork/yx306/RDR/JRSSB/CI/agp_icfm_712_merged_h20_20260920/agp_icfm_merged_ci.png), [split comparison](/cwork/yx306/RDR/JRSSB/CI/agp_icfm_712_merged_h20_20260920/split_calibration.png), [all candidate intervals](/cwork/yx306/RDR/JRSSB/CI/agp_icfm_712_merged_h20_20260920/candidate_intervals.csv), [selected intervals](/cwork/yx306/RDR/JRSSB/CI/agp_icfm_712_merged_h20_20260920/cell_intervals.csv), and [protocol](/cwork/yx306/RDR/JRSSB/CI/agp_icfm_712_merged_h20_20260920/protocol.json) retain the numerical evidence.

The 18 shared tests pass, including the eight original tests and ten new checks of merging, exact Gaussian multiplier dependence, full-family adjustments, degenerate cases, and exact enumerated C.2 coverage after data-dependent selection. An independent AGP audit reproduced all 210 candidates, all 10,000 bootstrap maxima, all 4,091 test attachments, and all 24 descriptive plot points. Clopper-Pearson bounds were checked by binomial-tail root inversion; the maximum transformed endpoint difference was $1.94\times10^{-13}$.

AGP remains a retrospective analysis with repeated samples within some subjects and historical preprocessing/provenance limitations. These are nominal intervals under the specified sampling assumptions, not measured AGP coverage rates or individual-RDR intervals.

```bash
python3 -B experiments/AGP/AGP_ICFM_merged_ci.py --min-count 20
python3 -B -m pytest -q tests/test_calibration.py tests/test_merged_calibration.py
```

Use a fresh `--output-dir` for another run. [AGP_ICFM_merged_ci.py](../../AGP/AGP_ICFM_merged_ci.py) also saves the executable source snapshot and a replay command in the report.

## Original fixed-partition algorithms

The following C.1 and C.2 definitions are retained for historical reproducibility. Their partition is fixed before observing calibration data. Their original critical values and guarantees should not be applied directly after calibration-dependent merging; use the extension above for that case.

## Shared inputs and setup

Inputs are a trained predictor $\hat r_{\mathrm{tr}}:\mathcal X\to[0,2]$, a partition $\{I_{n,j}\}_{j=1}^{K_n}$ of the score range, independent calibration observations $X_1,\ldots,X_n\sim P$ and $Y_1,\ldots,Y_m\sim Q$, and confidence level $1-\alpha_{\mathrm{CI}}$.

The calibration samples must be independent of each other and of network training. Fix the score-bin boundaries before examining calibration observations. Boundaries may depend on training data; equal-width bins are admissible. Use a consistent endpoint convention so every score belongs to exactly one bin.

Define the input-space cells by $A_{n,j}=\{z:\hat r_{\mathrm{tr}}(z)\in I_{n,j}\}$, and let $j_n(x)$ denote the cell containing a query $x$. Compute

$$
k_{P,n,j}=\sum_{i=1}^{n}\mathbf1\{X_i\in A_{n,j}\},
\qquad
k_{Q,n,j}=\sum_{i=1}^{m}\mathbf1\{Y_i\in A_{n,j}\}.
$$

The population target in a positive-mass cell is

$$
\theta_{n,j}
=\frac{2P(A_{n,j})}{P(A_{n,j})+Q(A_{n,j})}
=\mathbb E_{\widetilde Q}\{r_0(Z)\mid Z\in A_{n,j},\mathcal D_{\mathrm{tr}}\},
\qquad \widetilde Q=(P+Q)/2.
$$

Both procedures quantify **calibration-sample uncertainty about this cell average**, conditional on the trained network and partition. They do not estimate variation of the original network prediction across training runs, or variation of the true RDR within a cell. All queries in the same cell receive the same result.

## Algorithm C.1: asymptotic marginal confidence interval

**Goal:** construct an interval for the population RDR of the cell containing a fixed query $x$.

### 1. Compute the calibrated estimate

For $j=j_n(x)$, write $\hat p=k_{P,n,j}/n$ and $\hat q=k_{Q,n,j}/m$. If $\hat p+\hat q>0$, set

$$
\hat r_{\mathrm{cal},n}(x)=\frac{2\hat p}{\hat p+\hat q}.
$$

Use separate sample proportions. The raw-count ratio $2k_{P,n,j}/(k_{P,n,j}+k_{Q,n,j})$ is valid only when $n=m$.

### 2. Compute the standard error

$$
\widehat{\operatorname{se}}_{n,m}^{\,2}(x)
=
\frac{4\hat q^{\,2}\hat p(1-\hat p)}{n(\hat p+\hat q)^4}
+
\frac{4\hat p^{\,2}\hat q(1-\hat q)}{m(\hat p+\hat q)^4}.
$$

Take the square root. The formula already includes the sample-size factors; do not divide by another square root of sample size.

### 3. Construct the interval

Let $z_{1-\alpha_{\mathrm{CI}}/2}$ be the standard normal quantile. Return

$$
\mathcal I_{n,m}(x)
=
\left[
\max\{0,\hat r_{\mathrm{cal},n}(x)-z_{1-\alpha_{\mathrm{CI}}/2}\widehat{\operatorname{se}}_{n,m}(x)\},
\min\{2,\hat r_{\mathrm{cal},n}(x)+z_{1-\alpha_{\mathrm{CI}}/2}\widehat{\operatorname{se}}_{n,m}(x)\}
\right].
$$

**Exceptional cases:** if both counts are zero, set the estimate to $1$, the stored standard error to $0$, and the interval to $[0,2]$. If the cell is nonempty but the estimated standard error is zero, retain the ratio estimate and return $[0,2]$. Flag either case as an unavailable normal approximation. Do not add pseudocounts.

**Output:** cell index, calibrated estimate, standard error, interval endpoints, and any exceptional-case flag.

**Theory and interpretation:** Theorem C.1, under Assumptions C.1–C.2, gives asymptotic marginal coverage $1-\alpha_{\mathrm{CI}}$ for $\theta_{n,j_n(x)}$. These assumptions include $m/n\to\rho\in(0,\infty)$ and, for some fixed $c\in(0,1/2)$, an interior cell ratio $c\leq\theta_{n,j_n(x)}\leq2-c$, mixture mass $0<\mu_n=\widetilde Q(A_{n,j_n(x)})\leq1-c$, and $n\mu_n\to\infty$, in the training-probability sense stated in the draft. The interval is neither finite-sample exact nor simultaneous across cells.

Corollary C.1 transfers coverage to the individual $r_0(x)$ only under Assumption C.4: $|\theta_{n,j_n(x)}-r_0(x)|/\operatorname{se}_{n,m}(x)\to_p0$.

## Algorithm C.2: finite-sample simultaneous confidence band

**Goal:** construct intervals covering all positive-mass cell RDRs simultaneously.

### 1. Allocate the error probability

Set $\delta=\alpha_{\mathrm{CI}}/(2K_n)$. Use the full prespecified number of bins, including empty bins. Each probability interval below has two tails of size $\delta/2=\alpha_{\mathrm{CI}}/(4K_n)$.

### 2. Compute binomial confidence limits

For $k$ successes among $N_0$ observations, define the Clopper–Pearson limits

$$
L_{\mathrm{CP}}(k,N_0;\delta)=
\begin{cases}
0,&k=0,\\
q_{\mathrm{Beta}}(\delta/2;k,N_0-k+1),&k>0,
\end{cases}
$$

$$
U_{\mathrm{CP}}(k,N_0;\delta)=
\begin{cases}
1,&k=N_0,\\
q_{\mathrm{Beta}}(1-\delta/2;k+1,N_0-k),&k<N_0.
\end{cases}
$$

Here $q_{\mathrm{Beta}}(u;a,b)$ denotes the $u$-quantile of a Beta distribution with parameters $a,b$.

For every cell $j$, apply these formulas with $(k,N_0)=(k_{P,n,j},n)$ to obtain $[L_{P,n,j},U_{P,n,j}]$, and with $(k,N_0)=(k_{Q,n,j},m)$ to obtain $[L_{Q,n,j},U_{Q,n,j}]$.

### 3. Transform the probability limits into RDR limits

Return

$$
\mathcal B_{n,m,j}
=
\left[
\frac{2L_{P,n,j}}{L_{P,n,j}+U_{Q,n,j}},
\frac{2U_{P,n,j}}{U_{P,n,j}+L_{Q,n,j}}
\right].
$$

For a query $x$, report $\mathcal B_{n,m}(x)=\mathcal B_{n,m,j_n(x)}$. Both denominators are positive when $n,m\geq1$. Empty cells automatically receive $[0,2]$; do not discard them or change $K_n$ after seeing the counts.

**Output:** interval endpoints for every bin and a query-to-bin lookup. The calibrated estimates from C.1 may also be reported. This algorithm uses neither an estimated standard error nor a normal quantile.

**Theory and interpretation:** Theorem C.3 uses Clopper–Pearson coverage, a union bound, and monotonicity of the RDR transformation to give

$$
\mathbb P\{\theta_{n,j}\in\mathcal B_{n,m,j}\text{ for every }j
\text{ with }\widetilde Q(A_{n,j})>0\mid\mathcal D_{\mathrm{tr}}\}
\geq1-\alpha_{\mathrm{CI}}.
$$

This is a finite-sample guarantee under the independence and fixed-partition requirements of Assumption C.1; no asymptotic sample-size ratio or Assumptions C.2–C.4 are needed. The band can be conservative, especially for sparse cells. It covers the calibrated population function simultaneously, **not the individual values of $r_0$ throughout each cell** without additional uniform bias control.
