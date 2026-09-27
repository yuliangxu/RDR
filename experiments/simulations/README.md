# Reproducible simulations

The retained workflow is **toy illustrations → model selection → matched loss
comparison → four-loss sample-size convergence at the locked architecture**. Completed results and the
local-gap definition are in [docs/simulations.md](../../docs/simulations.md).
Small numerical evidence is included under
[results/simulations](../../results/simulations/README.md).

## Retained files

| File | Purpose |
| --- | --- |
| `toy_illustrations.ipynb` | Gaussian tail separation and beta-mixture toy examples. |
| `population.py`, `config.json` | Five populations, analytic truth, seeds and full experimental design. |
| `workflow.py` | Common selection, fitting primitives, fit verification and diagnostic reports; original Hellinger-only convergence remains for compatibility and smoke checks. |
| `report.py` | Compact selection tables, supplementary results, Markdown/HTML/LaTeX and rendered tables. |
| `loss_followup.py` | Matched loss fitting, copied Hellinger references, and verified presentation replay. |
| `convergence.py`, `convergence_report.py` | Active four-loss convergence, frozen selection checks, compact tables and sample-size curves. |
| `selection.slurm`, `loss_followup.slurm`, `convergence.slurm` | Optional CPU array/report wrappers using saved source snapshots. |
| `computation_cost.py`, `cost_report.py`, `computation_cost.slurm` | Matched training-time replay, verified cost tables and a combined convergence/cost report. |
| `requirements.txt` | Recorded scientific environment. |

Shared objectives, architectures, fitting and diagnostics live in `utils`.
The old architecture/coverage replay module and its tests have been retired.
No datasets need downloading: all simulation observations are generated from
specified distributions. All output directories are configurable.

## Environment

Python 3.9.25 and the versions in [requirements.txt](requirements.txt) were used:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r experiments/simulations/requirements.txt
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
```

CPU fitting is the validated setting, with one PyTorch thread and deterministic
algorithms. The completed runs used PyTorch `2.6.0+cu124`, even though fitting
ran on CPU. Reusing those archived fits requires their exact recorded versions;
a different build suffix fails the intentional compatibility checks. A fresh
run records its own environment. The installed environment has been exercised;
a fresh dependency installation has not yet been tested.

Commands run from the repository root. Prepared runs save their scientific
source trees; use those snapshots for fitting/freezing/convergence, so later
checkout changes cannot silently change an existing experiment. Use a new
output directory for every new protocol or presentation.

## Quick execution check

This tiny example exercises selection, follow-up and convergence without running
the production grid. The compact selection renderer requires the full design,
so use the workflow's diagnostic report for this check.

```bash
python -B -m experiments.simulations.workflow selection --output runs/smoke-selection --smoke --prepare-only
python -B runs/smoke-selection/source/experiments/simulations/workflow.py selection --output runs/smoke-selection --smoke
python -B runs/smoke-selection/source/experiments/simulations/workflow.py freeze --output runs/smoke-selection
python -B runs/smoke-selection/source/experiments/simulations/workflow.py report --output runs/smoke-selection
python -B -m experiments.simulations.loss_followup prepare --reference runs/smoke-selection --output runs/smoke-loss
python -B runs/smoke-loss/source/experiments/simulations/loss_followup.py fit --output runs/smoke-loss --task-index 0
python -B runs/smoke-loss/source/experiments/simulations/loss_followup.py report --output runs/smoke-loss
python -B -m experiments.simulations.convergence prepare --reference runs/smoke-selection --output runs/smoke-convergence --repeats 2 --sample-sizes 12 24
for task in $(seq 0 3); do
  python -B runs/smoke-convergence/source/experiments/simulations/convergence.py fit --output runs/smoke-convergence --task-index "$task"
done
python -B runs/smoke-convergence/source/experiments/simulations/convergence.py report --output runs/smoke-convergence
```

These are implementation checks, not paper evidence. The toy notebook can be
executed separately from top to bottom; its optional export directory is
configurable. All six toy fits have also been executed successfully through a
Jupyter kernel using `nbclient` from a clean source copy.

## Reproduce model selection

The default grid has 23 distinct candidates over five settings and ten
repetitions: **1,150 fits**. Four losses are compared at MLP32/slope 2; Hellinger
also receives a five-architecture × four-slope search, with the shared baseline
counted once. Each distribution supplies 1,000 training, 500 validation, 5,000
calibration and 10,000 evaluation observations. Roles have disjoint random
streams, and candidates within a repetition share observations.

```bash
python -B -m experiments.simulations.workflow selection --output runs/selection --prepare-only
python -B runs/selection/source/experiments/simulations/workflow.py selection --output runs/selection
python -B runs/selection/source/experiments/simulations/workflow.py freeze --output runs/selection
python -B runs/selection/source/experiments/simulations/workflow.py report --output runs/selection
python -B -m experiments.simulations.report --run runs/selection --output runs/selection-publication
```

`freeze` selects one Hellinger activation/architecture combination by equally
weighted mean validation Brier. It does not select the best loss. Each fit
restores its minimum own-validation-loss checkpoint before cross-model Brier is
computed. The full grid must be complete and intact before freezing/reporting.
The completed study selected B8-Res64/slope 2.

## Reproduce the matched loss comparison

Prepare against the completed selection, then run all 50 tasks. Each task fits
KL, chi-square and JS for one case/repetition; 50 Hellinger fits are copied and
verified from the selected configuration. This adds **150 new fits**.

```bash
python -B -m experiments.simulations.loss_followup prepare --reference runs/selection --output runs/loss-followup
for task in $(seq 0 49); do
  python -B runs/loss-followup/source/experiments/simulations/loss_followup.py fit --output runs/loss-followup --task-index "$task"
done
python -B runs/loss-followup/source/experiments/simulations/loss_followup.py report --output runs/loss-followup
```

The wrapper preserves all data streams, initializations, optimizer settings and
loss-specific checkpoint rules. The report verifies 150 new and 50 reference
fits before writing `FOLLOWUP_COMPLETE.json`. It produces per-fit/aggregate
CSV files, compact rendered Markdown tables, HTML, LaTeX and vector PDFs, with
metric-specific winner bolding, paired differences and the local-gap definition.
This follow-up supplies additional design evidence after inspection of the
original selection results. It neither changes the frozen model nor launches
convergence.

## Regenerate reports from completed fits

The selection renderer verifies fitted inputs and writes a new presentation:

```bash
python -B -m experiments.simulations.report --run /path/to/completed-selection --output runs/selection-rerender
```

The follow-up renderer verifies archived sources and uses the frozen scientific
implementation to verify all fits before applying the current presentation:

```bash
python -B -m experiments.simulations.loss_followup render --reference /path/to/completed-followup --output runs/loss-rerender
```

This changes no fitted models, source archives, or original reports. The new
`rendering_manifest.json` records consumed artifacts, current rendering code and
versions. A presentation is not a new fitting run. Do not call the current
strict `report` command on an old follow-up whose saved reporting source differs
from the checkout; use `render` or that run's own frozen entrypoint.

## Four-loss sample-size convergence

The active convergence study compares **Hellinger, KL, chi-square and JS** at
the frozen **B8-Res64 architecture and output slope 2**. Its default design is
five settings × 100 repetitions × six training counts
`{100,200,500,1000,2000,5000}` per distribution × four losses:
**12,000 new fits**, organized into 3,000 case/repetition/sample-size tasks.
Each distribution still supplies 500 validation, 5,000 calibration and 10,000
evaluation observations. This varies training size while the other sample
budgets remain fixed. The production study is **complete: all 12,000 fits and
their reports are verified**. Together with the 1,300 distinct selection and
matched-loss fits, the retained studies contain **13,300 distinct fitted models**.
The separate 1,200 timing replays do not add independent scientific repetitions.

The [simulation summary](../../docs/simulations.md) and
[included convergence evidence](../../results/simulations/README.md) contain
the completed results. The [verification record](../../results/simulations/completed_studies_verification.json)
retains completion checks, and [provenance](../../results/simulations/provenance.json)
records original run locations and source hashes. Separate execution preflights
are not counted in the production grid.

Equal-setting mean MSE falls by approximately 85–90% from n=100 to n=5,000
across the four losses. KL and JS have similar mean MSE at n=5,000;
Hellinger has the largest mean MSE under this fixed architecture and training
recipe. Per-setting results and paired uncertainty are retained in the reports.

Convergence uses a fresh random namespace, independent of selection and the
matched-loss follow-up. Within a setting/repetition, initial network parameters
and validation/calibration/evaluation draws are shared across losses and sizes.
Training draws are shared across losses at each size and independently seeded
by size, so they are not nested. Sample roles remain independent. The optimizer,
scheduler, epoch budget and own-validation-loss checkpoint rule are unchanged;
architecture and activation are not retuned for each loss or sample size.

```bash
python -B -m experiments.simulations.convergence prepare --reference runs/selection --output runs/convergence
for task in $(seq 0 2999); do
  python -B runs/convergence/source/experiments/simulations/convergence.py fit --output runs/convergence --task-index "$task"
done
python -B runs/convergence/source/experiments/simulations/convergence.py report --output runs/convergence
```

Preparation verifies the original selection records, scientific sources and
environment, then saves a complete 17-file source snapshot and checksummed
protocol. Every fit/report command verifies its executing source against that
snapshot. The report requires all 12,000 intact fits and writes
`CONVERGENCE_COMPLETE.json` only after checking all generated artifact hashes.
An incomplete grid cannot be reported, and a partial presentation cannot be
overwritten. Tables show the observed winner at each size, curves retain all
four losses, and paired-comparison CSVs use within-repetition differences.

The original `workflow.py convergence` entrypoint is retained as a
Hellinger-only compatibility and smoke helper. Use `convergence.py` for the
active four-loss study.

## Average computation cost

The original convergence protocol did not record per-fit durations. Its grouped
Slurm timings cover multiple sizes and losses, so cost is measured in a separate
replay of repetitions 0–9 from that same frozen design. This adds 1,200 timed
fits: five settings × ten repetitions × six sizes × four losses. It does not
change or augment the 12,000 accuracy fits. Training metadata, histories and
model parameters must match the corresponding reference fit before costs can
be published.

```bash
python -B -m experiments.simulations.computation_cost prepare --reference runs/convergence --output runs/convergence/computation_cost --repeats 10
for task in $(seq 0 299); do
  python -B runs/convergence/computation_cost/source/experiments/simulations/computation_cost.py fit --output runs/convergence/computation_cost --task-index "$task"
done
python -B runs/convergence/computation_cost/source/experiments/simulations/computation_cost.py report --output runs/convergence/computation_cost
```

The reference accuracy report must be complete before the final cost report.
For Slurm, use `computation_cost.slurm OUTPUT fit 6` with array `0-49%4`, then
`computation_cost.slurm OUTPUT report` after both the cost array and the original
convergence report succeed. A timing task warms each loss on throwaway data,
then rotates loss order and reinitializes actual models with their frozen seeds.
Timing includes the training call, validation and checkpoint restoration, and
excludes data generation, model initialization, evaluation/CI diagnostics,
I/O and queue waits. Record wall seconds and process CPU seconds separately;
one CPU and one numerical-library thread are used. Hardware, environment and
worker concurrency accompany the mean ± SD tables. These are 10-repetition
warmed training benchmarks, not durations recovered from the 100-repetition
accuracy experiment.

The [completed benchmark](../../docs/simulations.md#computation-cost) contains
**1,200 verified timing replays**, with [per-setting means and sample SDs](../../results/simulations/computation_cost/cost_summary.csv).
Mean training wall time is about four seconds per fit at n=1,000 and 12–18
seconds across settings/losses at n=5,000, using one numerical thread on an
Intel Xeon Gold 6226 CPU. Costs include data-dependent stopping. All timing
replays matched the original parameters, histories and training metadata exactly.

Choose the convergence directory's permanent location before preparing costs.
The frozen cost protocol records its absolute `reference_path`, which must remain
available for exact replay and reporting. To work at a new location, prepare a
new cost run against the convergence directory there; editing a prepared
protocol invalidates its checksums. This restriction does not affect fresh
reproduction at a user-selected path.

## Optional cluster execution and verification

Prepare once before launching concurrent workers. Selection task indices are
0–49, each fitting all 23 candidates; matched-loss indices are 0–49, each fitting
three losses. Four-loss convergence indices are 0–2999, each fitting all four
losses at one case/repetition/sample size. A task can be selected with
`--task-index`; completed intact fits are skipped, while partial or changed fits
are rejected. Custom selection designs use `--config` consistently on preparation
and fitting. The supplied `selection.slurm` wrapper runs the default production
configuration only; for smoke or custom designs, use the direct Python commands
and pass the matching `--smoke` or `--config` arguments. Fixed 20-bin CIs are
default; `adaptive: true` adds the specified h=100 merged-cell diagnostic.

All Slurm wrappers accept an absolute prepared output directory and request
one CPU, 4 GB and two hours per task. Submit `fit` as array `0-49%10`, then an
`afterok` job for `selection.slurm OUTPUT finalize` or
`loss_followup.slurm OUTPUT report`. Supply the site's partition, account and
log paths to `sbatch`. Every worker executes the saved source snapshot.

For convergence, group the six sizes of each case/repetition in one Slurm
array unit: **`0-499%20`**, with `convergence.slurm OUTPUT fit 6`. Each unit
runs six logical tasks (24 fits); `6` must match the number of prepared sample
sizes. Submit `convergence.slurm OUTPUT report` with an `afterok` dependency
on the whole array. For example, after replacing the site account and partition:

```bash
rdr_convergence_output="$(realpath runs/convergence)"
export RDR_PYTHON="$(command -v python)"
mkdir -p "$rdr_convergence_output/logs"
rdr_convergence_array=$(sbatch --parsable --partition=YOUR_PARTITION --account=YOUR_ACCOUNT --array=0-499%20 --output="$rdr_convergence_output/logs/fit-%A_%a.out" --error="$rdr_convergence_output/logs/fit-%A_%a.err" experiments/simulations/convergence.slurm "$rdr_convergence_output" fit 6)
sbatch --partition=YOUR_PARTITION --account=YOUR_ACCOUNT --dependency="afterok:${rdr_convergence_array}" --kill-on-invalid-dep=yes --mem=8G --output="$rdr_convergence_output/logs/report-%j.out" --error="$rdr_convergence_output/logs/report-%j.err" experiments/simulations/convergence.slurm "$rdr_convergence_output" report
```

```bash
python -B -m pytest -q -p no:cacheprovider tests
```

Tests cover scientific targets, noisy observed-space truth, split/seed pairing,
checkpoint restoration, source/artifact tampering, archived report rendering,
and shared fixed/adaptive CI calculations. Scientific sources and environment
requirements remain versioned with each run; compatibility checks are strict.

Verification includes clean-copy execution, a real 16-fit/two-size convergence
smoke report, exact comparison with the original Hellinger implementation, and
timing-runner checks for measured boundaries and exact reference replay.
The release check passed all **105 tests** from a clean source copy in the
recorded scientific environment. Git attributes preserve checksummed source
and evidence bytes when cloning on systems with different line-ending defaults.
Re-rendering the completed 200-model follow-up reproduced all 24 published
artifacts byte for byte, with 1,460 consumed file hashes verified. See the
[evidence verification record](../../results/simulations/completed_studies_verification.json)
for completed-study checks and their limits.
