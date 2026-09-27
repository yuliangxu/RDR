# Simulation evidence

These CSV files, protocols and publication figures support
[docs/simulations.md](../../docs/simulations.md). They contain about 9 MB of
numerical evidence and figures, without checkpoints or raw predictions.

| Folder | Completed design |
| --- | --- |
| `model_selection/` | Five settings, ten repetitions, 23 candidates: 1,150 fits. |
| `loss_comparison/` | Four losses at B8-Res64, slope 2: 150 new fits plus 50 reused Hellinger fits. |
| `convergence/` | Four losses × five settings × six training sizes × 100 repetitions: 12,000 fresh fits. |
| `computation_cost/` | Exact replays of ten convergence repetitions per setting/size/loss: 1,200 timed fits. |

There are 13,300 distinct scientific fits. The reused Hellinger rows and timing
replays are not additional independent scientific repetitions.

For scientific studies, `metrics.csv` contains per-fit MSE/Brier values and
`calibration.csv` contains per-fit local gaps, widths and supported mass.
`summary.csv` gives per-setting means and Monte Carlo standard errors;
`aggregation.csv` weights the five settings equally. `paired_comparisons.csv`
uses matched within-repetition differences. `protocol.json` records each design.
Convergence files retain sample size as a separate factor.

For computation cost, `cost_results.csv` contains all measured wall/CPU seconds,
epochs and per-fit provenance; `cost_summary.csv` reports arithmetic means and
**sample SDs**, not Monte Carlo SEs, across ten timing repetitions per cell.
`cost_hardware.json` records the observed CPU, host and thread settings. Training
timers include validation and checkpoint restoration, excluding data generation,
initialization, evaluation/CI diagnostics, I/O and queue waiting.

The convergence [overall curves](convergence/figures/convergence_overall.png) and
[per-setting curves](convergence/figures/convergence_by_setting.png) have vector
PDF counterparts. Main accuracy tables and five per-setting wall-time tables
are retained as PNG/PDF figures and LaTeX. Additional per-setting accuracy,
CPU-time, epoch and parameter-count tables are available in the corresponding
`appendix_tables.tex` or `cost_appendix_tables.tex` file. These artifacts use
relative paths and are readable without access to the original HPC runs.

[provenance.json](provenance.json) records file hashes, versions, source hashes,
completion identities and original run locations.
[completed_studies_verification.json](completed_studies_verification.json)
records the complete-grid numerical audits and successful Slurm accounting for
500 convergence units, 50 timing units and two report jobs. Scientific means and
Monte Carlo SEs, paired contrasts, and timing means/SDs were independently
recomputed from the included per-fit records. The refresh also verified all
161 published report-artifact paths and 19 source files. An additional reread
of every original fitted file was stopped after at least 7,000 verified fits
when storage reads stalled; this limitation is recorded separately from the
full fit and exact-replay checks performed by the completed report jobs.

Use the [reproduction commands](../../experiments/simulations/README.md) to
regenerate the studies and reports from synthetic data at any chosen output
location. Rendering from existing fitted studies requires their full artifact
directories and recorded environment; the included CSVs provide portable
numerical evidence, without checkpoints or predictions. Original full-report
locations are recorded in [provenance.json](provenance.json).
