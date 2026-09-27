# AGP real-versus-ICFM calibration intervals

## Architecture comparison with validation RDR and CIs

[AGP_architecture_diagnostics.py](AGP_architecture_diagnostics.py) runs 15 fits: `baseline_p5`, `baseline_p30`, `wide_p30`, `deep_p30`, and `residual_p30`, each with seeds 20260918, 20260919, and 20260920. The narrow controls separate stopping patience from capacity; the deep and residual models share their linear layers and activation placement, differing by the skip connections. All fits use the same smoothed ILR representation ($\epsilon=10^{-6}$), separate-mean Hellinger objective, bounded output $2\operatorname{sigmoid}(2a)$, archived observations, and split assignments. Training contains 9,162 P / 7,310 Q observations, validation 1,281 P / 1,044 Q, and test 2,002 P / 2,089 Q.

Each checkpoint minimizes validation Hellinger loss. Validation balanced Brier selects the overall run and a representative seed within each method before test scoring. The [experiment report](/cwork/yx306/RDR/JRSSB/CI/agp_architecture_20260925/report.md) includes [training/validation/test score-versus-frequency diagnostics](/cwork/yx306/RDR/JRSSB/CI/agp_architecture_20260925/score_vs_frequency.png), [selected-model merged CIs](/cwork/yx306/RDR/JRSSB/CI/agp_architecture_20260925/selected_model_ci.png), and [selected-model elementary-bin CIs](/cwork/yx306/RDR/JRSSB/CI/agp_architecture_20260925/selected_model_elementary_ci.png). It reports all seeds, balanced score-region masses, and conditional signed and absolute gaps for the fixed middle range [0.2,1.8), alongside global gaps, Brier, and AUC. Validation curves are descriptive because validation influences fitting and selection; no validation CIs are claimed.

Test CIs use $h=40$ P and Q observations per merged region, nominal 95% confidence, all 210 contiguous unions of the 20 elementary bins, and 10,000 Gaussian-multiplier repetitions for C.1. C.2 uses candidate-adjusted Clopper–Pearson bounds. The elementary-bin figure reuses the same candidate-adjusted intervals. Protection is per fitted model, not joint across the 15 networks. Numeric cutoffs are shared, but each network induces different input-space cells. The intervals concern cell-average RDR; reused test observations and repeated P observations within subjects retain the retrospective nominal interpretation and sampling limitations described below.

In the completed comparison, all 15 merged partitions collapse to the whole score range. Their structural interval $[1,1]$ provides no regional calibration evidence; the elementary-bin figure preserves the sparse middle-region uncertainty. Validation Brier selects `deep_p30_s20260920`, but its test Hellinger objective worsens because of an extremely small score on a real P observation. The report also compares the residual model's smaller descriptive middle-region gaps, retaining the original validation selection.

For the validation-selected residual fit (`residual_p30_s20260919`, epoch 25), the [equal-width CI report](/cwork/yx306/RDR/JRSSB/CI/agp_residual_equal_width_ci_20260925/report.md) and [three-panel plot](/cwork/yx306/RDR/JRSSB/CI/agp_residual_equal_width_ci_20260925/residual_mlp_equal_width_ci.png) reproduce the original AGP display with 20 bins of width 0.1. This uses original C.1 marginal intervals and C.2 simultaneous adjustment across 20 bins, distinct from the architecture report's 210-candidate intervals. [AGP_residual_equal_width_ci.py](AGP_residual_equal_width_ci.py) reuses the frozen test scores without fitting or merging; its output includes PDF, cell tables, and provenance.

The same report also includes [adaptive CIs with $h=20$](/cwork/yx306/RDR/JRSSB/CI/agp_residual_equal_width_ci_20260925/adaptive_h20/residual_mlp_adaptive_ci.png), added by [AGP_residual_adaptive_ci.py](AGP_residual_adaptive_ci.py). Adjacent merging gives two regions, [0,0.7) and [0.7,2], with P/Q counts 21/2,052 and 1,981/37. Adaptive C.1 uses the joint multiplier bootstrap, and C.2 corrects across all 210 candidate unions. These broad cell averages can conceal middle-score mismatch. The original equal-width artifacts are preserved, with the previous report and manifest archived under `report_revisions/before_adaptive_h20/`.

[AGP_residual_variance_ci.py](AGP_residual_variance_ci.py) adds true within-cell RDR variance upper confidence bounds to both partitions in that report. For each saved mean interval $[L,U]$, the upper bound is $c(2-c)$ with $c=\min\{U,\max\{L,1\}\}$, and the SD upper bound is its square root. [Equal-width](/cwork/yx306/RDR/JRSSB/CI/agp_residual_equal_width_ci_20260925/variance_bounds/equal_width/variance_upper_bounds.png) and [adaptive](/cwork/yx306/RDR/JRSSB/CI/agp_residual_equal_width_ci_20260925/variance_bounds/adaptive_h20/variance_upper_bounds.png) plots, augmented CSV tables, source snapshots, and a protocol are under `variance_bounds/`. Bounds inherit their source mean intervals' marginal/simultaneous validity and AGP sampling limitations. They quantify true within-cell dispersion, not the sampling variance of a mean estimate or a range for individual RDR values; see the [method derivation](../JRSSB/CI/agent_CI.md#upper-confidence-bounds-for-true-within-cell-rdr-variance).

To rerun into a fresh output directory:

```bash
python3 -B experiments/AGP/AGP_architecture_diagnostics.py --stage all \
  --output-dir /cwork/yx306/RDR/JRSSB/CI/agp_architecture_20260925_rerun \
  --seeds 20260918 20260919 20260920 --epsilon 1e-6 \
  --bootstrap-repetitions 10000
```

The output preserves source snapshots, checkpoints, transforms, split provenance, row-level scores, all candidate and selected-region CI tables, and checksums. `protocol.json` freezes the fitting configuration; `selection.json` and `report_protocol.json` record the completed validation selection. [AGP_architecture_report.py](AGP_architecture_report.py) regenerates figures and the report from the saved tables.

## Compositional-input model diagnostics

[AGP_composition_diagnostics.py](AGP_composition_diagnostics.py) compares the existing raw-input MLP, a closed-abundance MLP control, ILR + MLP, an exactly constrained learned log-contrast first layer, and phylogenetic ILR + MLP. It preserves the archived 7:1:2-policy samples, the three 32-unit hidden layers, bounded RDR output, Hellinger objective, and training schedule. The raw reference seed must reproduce the archived checkpoint exactly.

The primary comparison uses seeds 20260918, 20260919, and 20260920, with additive smoothing $\epsilon=10^{-6}$ after closure for the log-ratio models. Sensitivity fits use $10^{-7}$ and $10^{-5}$ at the first seed and report training/validation diagnostics only. The CLR center and scalar RMS are fitted on equally weighted P/Q training inputs. All log-ratio parameterizations begin with matching predictions; full ILR and the constrained contrast layer have the same function class but different optimization coordinates. [AGP_phylo_basis.py](AGP_phylo_basis.py) verifies the archived binary tree, generator taxon order, and orthonormal balances.

The [comparison report](/cwork/yx306/RDR/JRSSB/CI/agp_composition_20260922/report.md) includes score-versus-frequency curves, weighted absolute and RMS gaps, balanced Brier score, AUC, seed variation, and zero-replacement sensitivity. The architecture diagnostic retains 20 fixed numeric score bins; this is distinct from the $h=40$ merged CI analysis below. Each network defines different input-space score cells. Checkpoints use validation Hellinger loss; the primary run is selected by validation Brier before test predictions are computed. The previously inspected test cohort remains a retrospective comparison.

The completed primary comparison improves aggregate predictive metrics: the mean test absolute gap across three seeds is 0.1160 for raw inputs, 0.0501 for ILR, 0.0495 for learned log-contrasts, and 0.0625 for PhILR. This does not remove the middle-range score-versus-frequency discrepancy. More probability mass moves toward endpoint scores; the report separately shows bin masses and descriptive middle-range gaps, alongside sensitivity to zero replacement.

```bash
python3 -B experiments/AGP/AGP_composition_diagnostics.py \
  --output-dir /cwork/yx306/RDR/JRSSB/CI/agp_composition_20260922_rerun
```

Use a fresh output directory. Models, transform parameters, row-level scores, exact axis formulas, source snapshots, and checksums are saved with the report. Scientific invariant checks:

```bash
python3 -B -m unittest experiments.AGP.test_AGP_composition_diagnostics experiments.AGP.test_AGP_phylo_basis
```

## Adjacent-cell merging

[AGP_ICFM_merged_ci.py](AGP_ICFM_merged_ci.py) applies the shared merging extension in [agent_CI.md](../JRSSB/CI/agent_CI.md) to the frozen 7:1:2-policy checkpoint. The user selected **h = 20 P and 20 Q observations per merged region**. Scan the original 20 elementary bins from left to right, close each region once both counts reach h, and merge any leftover tail into the preceding region. All 210 contiguous candidate regions are protected before selecting the reported partition.

C.1 uses a joint Gaussian multiplier bootstrap over all candidates, with 10,000 repetitions and seed 20260920. It is an asymptotic simultaneous extension of the original marginal interval. C.2 uses Clopper-Pearson probability intervals with tail probability alpha/(4J), where J remains 210 after merging. Both methods use the same selected regions; their population targets are the averages within those merged regions. The structural whole-range target is known to be 1 and receives [1,1].

```bash
python3 -B experiments/AGP/AGP_ICFM_merged_ci.py --min-count 20
```

The [merged report](/cwork/yx306/RDR/JRSSB/CI/agp_icfm_712_merged_h20_20260920/report.md) includes all candidate intervals, the selected regions, test-sample attachments, bootstrap maxima, and common-region training/validation/test diagnostics. It preserves the checkpoint, scores, split manifest, and training history byte for byte. No fitting or generation is repeated. Use a fresh `--output-dir` for another run; the source snapshot and recorded command support reproduction. `--preflight` permits disposable output below `/tmp`; durable results must be below `/cwork/yx306/RDR`.

Merging increases cell counts while reducing score resolution. Width comparisons with the original analysis have different cell targets, and C.1 also changes from marginal to simultaneous inference. The sampling limitations at the end of this document still apply.

## Frozen fit and original 20-bin results

The [approximately 7:1:2 rerun](/cwork/yx306/RDR/JRSSB/CI/agp_icfm_712_20260920/report.md) refits the RDR after moving observations from validation to training. All eligible P data are retained, and the test/calibration observations are exactly the same as in the previous analysis. Q uses approximately 70/10/20; P's generator-unseen test pool limits its overall allocation to approximately 74/10/16, with the remaining fit pool divided approximately 7:1 by whole subjects.

| Current role | P samples | P subjects | P fraction | Q samples | Q fraction |
|---|---:|---:|---:|---:|---:|
| Training | 9,162 | 7,644 | 73.62% | 7,310 | 70.00% |
| Validation | 1,281 | 1,088 | 10.29% | 1,044 | 10.00% |
| Test = calibration | 2,002 | 1,967 | 16.09% | 2,089 | 20.00% |

```bash
python3 -B experiments/AGP/AGP_ICFM_712_ci.py --p-policy keep-all
```

Use a fresh `--output-dir` for a rerun. [AGP_712_data.py](AGP_712_data.py) performs the deterministic reallocation and verifies all source-row hashes, test IDs, and subject separation. The generator, network architecture, separate-mean Hellinger objective, initialization seed, optimizer, stopping rule, and 20 score bins match the prior fit. The 609 excluded real test rows remain excluded. An optional `--p-policy match-ratio` retains exactly 7,007/1,001/2,002 P rows by reserving 2,435 fit-pool samples, but that policy is not used for the current result.

The report includes the refitted test CIs, descriptive training/validation/test calibration curves, the selected checkpoint and loss history, and comparison against the previous model on identical test observations. Training and validation curves are not independent calibration evidence. New and old fitted maps define different score cells, so their interval widths are descriptive comparisons rather than precision comparisons for one unchanged target. To replay the saved checkpoint with verified inputs and code, add `--replay-from /cwork/yx306/RDR/JRSSB/CI/agp_icfm_712_20260920` and a fresh output directory.

## Previous full-test fit

The previous [full-test report](/cwork/yx306/RDR/JRSSB/CI/agp_icfm_full_test_20260918/report.md) uses **the same observations for test and calibration: 2,002 real P samples and 2,089 ICFM Q samples**. [AGP_ICFM_test_ci.py](AGP_ICFM_test_ci.py) verifies the completed fit below and combines its former calibration/query subsets. The fitted model, selected epoch 498, training/validation rows, and 20 score bins stay fixed. No retraining is needed for this historical replay.

```bash
python3 -B experiments/AGP/AGP_ICFM_test_ci.py \
  --fit-from /cwork/yx306/RDR/JRSSB/CI/agp_icfm_20260918 \
  --output-dir /cwork/yx306/RDR/JRSSB/CI/agp_icfm_full_test_20260918
```

Use a fresh output directory for a rerun. The full-test results retain sample provenance in `split_manifest.csv` and `scores.csv`, attach CIs in `test_intervals.csv`, and preserve the original fit protocol, checkpoint, and history. Both interval sets are compared using identical full-test weights: the equal-mixture mean C.2 width decreases from 0.31129 to 0.23015. There are no unavailable C.1 cells in the full-test result. C.1 and C.2 now have separate plots alongside the bin counts.

C.2 simultaneously covers the fixed population cell targets under its sampling assumptions, so attaching its intervals to the calibration observations does not require another query holdout. These same-sample summaries are not an independent calibration assessment. C.1 remains marginal for a fixed cell. The historical fit and its smaller-calibration results below remain available for reproduction.

[AGP_ICFM_ci.py](AGP_ICFM_ci.py) fits a separate relative-density-ratio model and applies the existing [C.1/C.2 algorithms](../JRSSB/CI/agent_CI.md) to real AGP samples versus the archived ICFM generator. The historical notebook and generator are preserved. The old RDR fit cannot supply independent calibration data because it trained on nominal test observations and reused their mixture in validation.

The new fit uses the original 614-taxon representation, three 32-unit hidden layers, bounded output, and midpoint Hellinger objective. For unequal sample sizes, the empirical loss is

$$
L=\frac12\overline{r_P^{-1/2}}+\frac14\overline{\sqrt{r_P}}+\frac14\overline{\sqrt{r_Q}}-1.
$$

Separate P/Q means preserve the population minimizer $r_0=2p/(p+q)$. The absolute minimum-validation checkpoint is restored; calibration/query data do not influence fitting or the 20 equal-width score bins. CIs target $2P(A_j)/(P(A_j)+Q(A_j))$, conditional on the frozen score map and partition.

[AGP_ci_data.py](AGP_ci_data.py) reconstructs original raw-data sample IDs and verifies them against the abundance table. It joins metadata by sample ID because the historical metadata order differs from abundance order. It excludes real test subjects represented in generator training and keeps each remaining subject within one role. Historical generated train/test banks use shared random starts; only the generated training bank is used and repartitioned.

| Role | P samples | P subjects | Q samples |
|---|---:|---:|---:|
| Training | 8,252 | 6,985 | 6,265 |
| Validation | 2,191 | 1,747 | 2,089 |
| Test = calibration (previous fit) | 2,002 | 1,967 | 2,089 |
| Previous calibration subset | 997 | 983 | 1,044 |
| Previous query subset | 1,005 | 984 | 1,045 |
| Excluded generator-training subjects from real test | 609 | 395 | — |

To reproduce the original fit and its former calibration/query split, run from the repository root:

```bash
python3 -B experiments/AGP/AGP_ICFM_ci.py
```

Default output: `/cwork/yx306/RDR/JRSSB/CI/agp_icfm_20260918/`. Existing outputs are rejected; set `--output-dir` to a fresh `/cwork` directory for a rerun. Full fitting uses at most 2,000 epochs, a 300-epoch minimum, validation patience 5, and two CPU threads. To replay the saved model into another directory:

```bash
python3 -B experiments/AGP/AGP_ICFM_ci.py \
  --replay-from /cwork/yx306/RDR/JRSSB/CI/agp_icfm_20260918 \
  --output-dir /cwork/yx306/RDR/JRSSB/CI/agp_icfm_20260918_replay
```

The [report](/cwork/yx306/RDR/JRSSB/CI/agp_icfm_20260918/report.md) contains all cell intervals. Artifacts include the model, training history, all sample scores, query intervals, inclusion/exclusion manifests, input/output hashes, source snapshots, and PNG/PDF plots. The saved source tree also supports replay with the same environment and source data.

These are retrospective nominal intervals, not measured coverage or individual-RDR intervals. C.1 is asymptotic; C.2's finite-sample statement assumes independent observations. Repeated samples within a subject, whole-table historical preprocessing, and incomplete historical generator split/latent manifests prevent an unconditional coverage claim. The real evaluation cohort consists of retained samples from subjects absent from generator training.
