# DCGAN model selection evidence

The [completed report](RESULTS.md) contains all 80 fits (four losses, four
sigmoid slopes, five paired repetitions) and five assessments after selection.
The registered rule selected JS with output `2 sigmoid(2z)`.

The selection Brier is 0.098344 and local gap is 0.081758. The selected model's
final-assessment Brier is 0.091056 and local gap is 0.066951, averaged across
five fitting repetitions. Chi-square/slope 0.5 has the smallest selection local
gap (0.060505), with higher Brier (0.104139); the chosen model does not minimize
both criteria. The full report preserves this trade-off and the middle-region
conditional discrepancies. Across-repetition SD describes fitting/generator
variation conditional on the fixed real data, not a population confidence interval.

These compact files copy the completed run at
`/cwork/yx306/RDR/MNIST/dcgan_model_selection_20260927/`. The
[manifest](manifest.json) records original locations, hashes, job IDs and the
lossless grouping of selected cell tables. All five selected-model calibration
tables are included for [selection](selected_selection_cells.json) and
[final assessment](selected_final_cells.json). Large checkpoints, score arrays,
raw data, all-candidate cell tables and frozen executable source remain at the
run root. They are not silently replaced by this compact report bundle.

[Independent verification](verification.json) checks all 85 assessments and
1,700 cells against the saved score arrays, including interval endpoints,
checkpoint selection, split disjointness and hashes. Brier recomputes exactly;
the largest discrepancy across the other checked numerical quantities is
below 9e-16.

Cell intervals target population cell-average RDR, not individual-image RDR.
The new run held final roles apart from fitting and selection, but the official
MNIST test images had already been inspected historically. Generator pretraining
membership remains unaudited. See the [procedure](../../../experiments/MNIST/README.md)
and [dataset summary](../../../docs/MNIST.md) for the precise protocol.
