# CelebA paper results

The expanded study with **one merged held-out test pool** is the main result.
The selected-model assessment and all five null families are complete. The
former final-calibration and final-evaluation observations now jointly supply
cell intervals, neural summaries, Brier/Gap and examples. Main counts are
39,829 P and 40,000 per Q; model fitting and selection are unchanged.
The loss/activation grid preserves one
documented numerical failure: 319/320 fits completed. Accepted models use JS
with `r = 2 sigmoid(0.5 z)` for features and `r = 2 sigmoid(2 z)` for pixels.
Architectures were fixed, and each comparison has five training repetitions.

Read [the scientific summary](../../docs/CelebA.md) for results, exact split
counts, interpretation and reproduction commands.

## Retained evidence

| Stage | Report | Tables and figures |
| --- | --- | --- |
| Expanded FID | [FID results](expanded_fid_20260929/FID_RESULTS.md) | [CSV](expanded_fid_20260929/FID_RESULTS.csv); all three pools complete |
| Loss/activation selection | [Selection results and failure caption](model_selection_expanded_20260929/RESULTS.md) | [Per-fit](model_selection_expanded_20260929/per_fit.csv), [per-candidate](model_selection_expanded_20260929/per_candidate.csv); feature/pixel heatmaps |
| Selected-model final assessment | [Merged-test results](merged_test_20261003/RESULTS.md) | [Four comparisons](merged_test_20261003/main_per_comparison.csv), [20 evaluations](merged_test_20261003/main_per_fit.csv) |
| Selected-model CIs and image bins | [CI and all-bin gallery](merged_test_20261003/RESULTS.md#all-figures) | Ten CI figures, eight compact image panels, [900 main/null cell records](merged_test_20261003/cell_intervals.csv), [exact displayed ranges](merged_test_20261003/image_groups.csv) |
| Publication labels and null histograms | [Updated gallery](publication_figures_20261003_v2/FIGURES.md) | Four three-digit image panels; [25-panel null overview](publication_figures_20261003_v2/null_rdr_histograms.pdf) and five family PDFs with 10-point labels at 7.2-inch width |
| Feature attribute associations | [Joint log(2−RDR) regressions](attribute_regression_expanded_20261003_v2/RESULTS.md) | [Rankings](attribute_regression_expanded_20261003_v2/attribute_summary.csv), [400 coefficients with identity-clustered intervals](attribute_regression_expanded_20261003_v2/coefficients_per_fit.csv), [plot](attribute_regression_expanded_20261003_v2/attribute_coefficients.pdf) |
| Attribute interactions and response sensitivity | [Held-out report](attribute_interactions_20261003/RESULTS.md) | [Accuracy](attribute_interactions_20261003/metrics_summary.csv), [deletion importance](attribute_interactions_20261003/importance_summary.csv), [response stability](attribute_interactions_20261003/response_stability.csv); three publication figures |
| Five learned null families | [Merged-test null results](merged_test_20261003/RESULTS.md#learned-null-controls) | [Five families](merged_test_20261003/null_per_comparison.csv), [25 evaluations](merged_test_20261003/null_per_fit.csv), Brier/RDR-error and CI figures |

There are **32 current figure pairs (PNG and PDF)**: two selection figures,
20 merged-test figures (including four revised image panels), six null histogram figures, and one feature-attribute coefficient plot, and three interaction/response-sensitivity figures. The compact ranked-image panels show the 40
globally smallest scores, closest to 1, and largest scores per source, with
actual displayed ranges at three significant digits. The [current image and histogram
gallery](publication_figures_20261003_v2/FIGURES.md) contains the presentation update;
full-precision image ranges and scores are preserved in its CSVs.
Repeat 00 is the fixed primary illustration; the remaining CI repetitions
and all-20-bin panels are supplements. Equal display capacity does not
represent population prevalence. Superseded figure revisions and split-based
reports are retained in
[RDR-working](https://github.com/yuliangxu/RDR-working/tree/celeba-working-20261003)
and the external research archive, rather than this final paper package.

Regression row-level design/response files remain under `/cwork`; coefficient
tables, diagnostics and the final plot are included. All 40 annotations jointly
explain about 3%/6% of the log-response variation for Q_l/Q_u; these are
exploratory associations.

All main and null per-evaluation `metrics.json` and `cells.json` records are
included. The known null truth is RDR 1 and optimal population Brier 0.25.
Null histograms overlay folds A/B, mark RDR 1, and shade the central 95% of
the equally weighted score distributions; this shading is not a confidence interval.
Null scores remain learned, with small residual errors; there was no exact-one
replacement, null-specific tuning, or pass/fail threshold.

## Integrity and reproduction

The [manifest](manifest.json) pins **302 artifacts (54.40 MiB)** with SHA-256
hashes, including exactly **32 PNG/PDF figure pairs**. Figures and numeric tables
are byte-identical to the archived runs. Three reports have explicitly recorded
link/label corrections so readers reach the accepted settings and latest plots;
no numeric result was changed. The checked-in
[export policy](../../experiments/CelebA/paper_files.json) also pins upstream
hashes and each text replacement.

Four compact `frozen_source.tar.gz` files preserve original FID, selection,
scoring and null pipeline versions. Later frozen sources remain in the complete
working snapshot and the external run directories; the active reproducible
pipeline is in [experiments/CelebA](../../experiments/CelebA/README.md).

Original completion/protocol receipts describe full archived runs. Some files
inside those receipts are external inputs or superseded artifacts deliberately
omitted here. The package manifest is the authoritative inventory for offline
verification. Root-level audit records document the earlier scientific checks;
[verification.json](verification.json) records checks of this final checkout.

Verify without datasets, GPUs or HPC access:

```bash
python3 -B experiments/CelebA/package_paper.py verify
```

Rebuild exactly the final artifacts into a fresh directory from the research
archive, with no intermediate historical export:

```bash
python3 -B experiments/CelebA/package_paper.py build \
  --source-root /cwork/yx306/RDR/CelebA --output /tmp/celeba-paper-export-NEW
python3 -B experiments/CelebA/package_paper.py verify --output /tmp/celeba-paper-export-NEW
```

The builder verifies every pinned upstream file and frozen source member before
writing, and refuses to overwrite changed files. It does not retrain models or
recompute FID. A source-root override supports a relocated archive with the same
relative run layout and exact content.

## Closeout scope

The full pre-cleanup snapshot was published first to RDR-working, commit
`6712ddc3cc0336182b5c89ed1af13036b56a2565`, on `main` and
`celeba-working-20261003`. A fresh clone matched all 949 tracked files. A separate
complete archive independently verified 4,437 files, including ignored files and
Git metadata. [CLEANUP.json](CLEANUP.json) records the archive and removed paths.
External datasets, fitted models and research runs remain intact.

Use these expanded results for the paper. Retain the failed selection cell and
its explanation: settings were explicitly accepted after reviewing the 319/320
grid, and the automatic selector did not complete. Five repetitions measure
training variation on fixed data. CIs are nominal retrospective image-level
cell-average diagnostics; FID closeness is descriptive. Attribute results are
exploratory associations, not causal drivers.

Global divergence/bootstrap contrasts, primary-comparison score histograms,
architecture sensitivity and a sixth real-feature null were not part of this
closeout. No additional experiment is needed unless the manuscript adds those
claims. See [REPRODUCTION.md](REPRODUCTION.md) for external inputs and the
remaining portability work before independent fresh-machine retraining.
