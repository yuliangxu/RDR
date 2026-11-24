# Retrospective AGP threshold experiment

This opt-in comparison asks whether access to taxon-specific abundance thresholds lowers held-out balanced Brier score for each of the four published AGP generators. Large private inputs, per-epoch histories, trained networks, and assessment predictions live on the research filesystem. The compact [results](../../results/AGP/relu_threshold_20261122/report.md) contain the four-generator comparison and links to per-seed scores.

## Estimators and splits

Each generator uses its original final training, stopping, and test arrays with 614 taxa. The test bank contains 2,002 real rows and 2,089 generated rows per generator. The run protocol records the six array digests for each pair and the sample-manifest digest. The clipped feedforward ReLU estimator divides each row by its sum, using the composition comparison's float64 normalization, then casts to float32. It takes each taxon's scaled abundance and trainable hinges at up to eight distinct positive training quantiles. Initial cutoffs are observed inverse empirical quantiles at 1/9 through 8/9 in the default configuration. The pooled training values determine the scale for each taxon, the largest of its positive 90th percentile, its maximum divided by 32, and 1e-8. The all-zero scale is one. Duplicate cutoffs collapse and cutoffs at the training maximum are omitted because they have no active training rows. An absent, singleton-positive, or constant-positive taxon keeps its linear feature.

The additive readout and the two width-128 ReLU interaction layers share these features. The final scalar is clipped to [0.01, 1.99]. Its intercept starts at one and both readouts have small random weights. For real and generated rows, write the two outputs as rP and rQ. Each epoch takes one AdamW step on the full equal-source loss
`-0.5 * mean(log(rP / 2)) - 0.5 * mean(log1p(-rQ / 2))`.
The trainer accumulates classwise sums in chunks and divides by each class's full count. It applies the logarithms directly to probabilities derived from the clipped output. The stopping Brier selects the minimum over all completed epochs. Histories record endpoint fractions separately for real and generated stopping rows. Saved state includes scales, the taxon index of each hinge, trainable cutoffs, output bounds, and weights.

The control uses the published sigmoid MLP on the square roots of closed abundances concatenated with binary presences. It retains the selected per-generator neural loss and sigmoid slope. Both new arms use the five original final initialization seeds. Each fit reads only the train and stopping banks and reloads its saved model to verify stopping predictions. Assessment opens the test banks after all 40 checkpoints are frozen.

## Commands

The commands below run from the RDR repository root in the Duke cluster environment. The recorded run used the following directory. For an independent refit, assign a fresh sibling directory to `agp_run` because preparation preserves existing runs.

```bash
agp_run=/cwork/yx306/RDR/AGP/relu_threshold_20261122
agp_report=results/AGP/relu_threshold_20261122
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
python3 -B -m pytest -q -p no:cacheprovider tests/test_agp_relu_threshold.py
python3 -B -m experiments.AGP.relu_threshold_experiment prepare --output "$agp_run"
sbatch --parsable --output="$agp_run/logs/%A_%a.out" experiments/AGP/relu_threshold.slurm "$agp_run"
python3 -B -m experiments.AGP.relu_threshold_baselines --output "$agp_run/baselines"
```

The original array was job `62437360`. Index 16 (Mixture + ICFM, fifth seed) failed the exact checkpoint-prediction replay check on an RTX 2080. The same frozen task was retried on the lab A5000 under job `62437863`, before test scoring.

```bash
sbatch --parsable --partition=mastatlab --array=16 --output="$agp_run/logs/%A_%a.out" \
  experiments/AGP/relu_threshold.slurm "$agp_run"
```

Once all fit tasks complete, score the stored checkpoints on CPU and construct the table.

```bash
for agp_index in {0..19}; do
  python3 -B -m experiments.AGP.relu_threshold_experiment assess \
    --output "$agp_run" --index "$agp_index" --device cpu || exit 1
done
python3 -B -m experiments.AGP.relu_threshold_report \
  --output "$agp_run" --report-dir "$agp_report"
```

Each array index equals four times the zero-based repeat plus the generator index, ordered as Mixture + ICFM, NADE + ICFM, MIDASim mask + ICFM, and Full MIDASim. The scheduler runs `fit --device cuda`. That command also accepts `--device cpu` and `--arm threshold` or `--arm control`. The compact per-seed file reports individual test Brier scores. Ensemble scores average the five predictions on each observation before scoring them.
