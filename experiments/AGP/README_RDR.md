# Reproducing the four-generator AGP evaluation

[AGP_RDR.md](../../docs/AGP_RDR.md) presents Bernoulli-mixture masks + ICFM,
NADE masks + ICFM, MIDASim masks + ICFM, and full MIDASim. This guide covers
the source and artifacts supporting those comparisons. The published results
reuse completed generator and RDR fits; publication did not rerun training.

## Environments and private inputs

Run commands from the repository root, inside suitable compute allocations.
[requirements.txt](requirements.txt) records the Python 3.9.25 evaluation
environment, including NumPy 1.26.4, SciPy 1.13.1, scikit-learn 1.6.1 and
PyTorch 2.6.0. It does not recreate the ICFM generator runtime. Generator
replay requires the archived runtime and checkpoints. MIDASim generation
used R 4.4.3 with the isolated MIDASim 2.0 installation; its launcher accepts
`MIDASIM_LIBRARY` for a separately installed library. ICFM/NADE fitting and
generation require GPU resources; RDR fits and downstream metrics use CPU.

```bash
export PYTHONPATH=/cwork/yx306/RDR/AGP/icfm_zero_20261007/runtime
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export MPLBACKEND=Agg MPLCONFIGDIR=/tmp/agp-rdr-reproduction-mpl
```

The current scripts pin the following completed private analysis roots:

| Generator | Analysis directory |
| --- | --- |
| Bernoulli mixture + ICFM | `/cwork/yx306/RDR/AGP/mask_rdr_20261012/analyses/bernoulli_mixture_icfm` |
| NADE mask + ICFM | `/cwork/yx306/RDR/AGP/mask_rdr_20261012/analyses/autoregressive_icfm` |
| MIDASim mask + ICFM | `/cwork/yx306/RDR/AGP/icfm_zero_20261007/analysis` |
| MIDASim | `/cwork/yx306/RDR/AGP/midasim_full_20261004/analysis` |

Required inputs include role-specific abundance arrays, sample/subject
manifests, frozen protocols, checkpoints, predictions and completion receipts.
The conventional classifier/generator archive is
`/cwork/yx306/RDR/AGP/mask_comparison_20261012`. Upstream preparation also
requires `/cwork/yx306/RDR/AGP/fixed_width_selection_20261003`,
`/cwork/yx306/RDR/AGP/icfm_corrected_20261006`, and the installed runtime under
`/cwork/yx306/RDR/AGP/midasim_pilot_20261004`. These are required dependencies
even though their earlier comparison panels are outside this paper.

The completed downstream archive is
`/cwork/yx306/RDR/AGP/four_generator_case_study_20261013`; the saved ordination
is `/cwork/yx306/RDR/AGP/four_generator_pcoa_20261014`. Public packages contain
aggregate tables, figures, source snapshots and hashes. They omit private
arrays, identities, row-level scores and model weights. A public checkout
alone therefore cannot rerun the private analyses. Relocating private inputs
requires an explicit path adaptation; `--output` changes destinations only.

## Check the public artifacts without private data

This checks the 82 files recorded in the six public package manifests using
only the Python standard library:

```bash
python3 -B - <<'PY'
from pathlib import Path
import hashlib, json
packages = (
    ('retained_mask_comparison_20261013', 'manifest.json'),
    ('four_generator_case_study_20261013', 'manifest.json'),
    ('four_generator_pcoa_20261014', 'manifest.json'),
    ('four_generator_pcoa_cp_20261015', 'manifest.json'),
    ('four_generator_class_abundance_20261022', 'manifest.json'),
    ('four_generator_class_abundance_compact_20261026', 'manifest_compact.json'),
)
checked = 0
for name, manifest_name in packages:
    root = Path('results/AGP') / name
    manifest = json.loads((root / manifest_name).read_text())
    for relative, expected in manifest['files_sha256'].items():
        actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(f'Hash mismatch: {root / relative}')
        checked += 1
print(f'Verified {checked} artifact hashes')
PY
```

Hash verification checks exported bytes, not numerical validity or training.
The published [case-study audit](../../results/AGP/verification_four_generator_20261013.json),
[ordination audit](../../results/AGP/verification_four_generator_pcoa_20261014.json)
and [CP-color audit](../../results/AGP/verification_four_generator_pcoa_cp_20261015.json)
record the earlier private numerical checks and their limitations.

## Replay downstream analyses from the completed fits

Choose previously unused destinations. The following commands read the
completed private studies and create new community/metric analyses; they do
not fit generators or neural RDR evaluators. Prepare both development-only
analyses before assessment. The shared community tree uses real stopping
subjects, which previously influenced RDR checkpoint selection.

```bash
AGP_CASE_OUTPUT=/cwork/yx306/RDR/AGP/four_generator_REPRODUCTION
AGP_CASE_PUBLIC=results/AGP/four_generator_REPRODUCTION
python3 -B experiments/AGP/four_generator_communities.py prepare --output "$AGP_CASE_OUTPUT/communities"
python3 -B experiments/AGP/four_generator_metrics.py prepare --output "$AGP_CASE_OUTPUT/metrics"
python3 -B experiments/AGP/four_generator_metrics.py assess --output "$AGP_CASE_OUTPUT/metrics"
python3 -B experiments/AGP/verify_four_generator_metrics.py --output "$AGP_CASE_OUTPUT/metrics"
python3 -B experiments/AGP/four_generator_communities.py assess --output "$AGP_CASE_OUTPUT/communities"
python3 -B experiments/AGP/four_generator_report.py \
  --communities "$AGP_CASE_OUTPUT/communities" --metrics "$AGP_CASE_OUTPUT/metrics" \
  --output "$AGP_CASE_PUBLIC"
python3 -B experiments/AGP/verify_four_generator_case_study.py \
  --root "$AGP_CASE_OUTPUT" --public "$AGP_CASE_PUBLIC" \
  --output "$AGP_CASE_OUTPUT/verification_with_public.json"
```

The report reuses `results/AGP/retained_mask_comparison_20261013` as its
opening RDR/conventional comparison. The publication version's metric
reference check uses the shipped four-generator global table; it does not
require a separate earlier metric package. Original source snapshots within
the sealed public packages remain unchanged. Fresh receipts contain new
paths/timestamps and are not expected to match original package hashes.

The PCoA producer reads the original sealed community assessment cache and
four completed RDR studies. It computes a fresh joint Bray–Curtis/Lingoes map
with all 10,358 observations. Request sufficient CPU/memory resources; the
included Slurm launcher requests four CPUs and 12 GB. Keep audit receipts
outside sealed plot directories:

```bash
AGP_PCOA_PRIVATE=/cwork/yx306/RDR/AGP/four_generator_pcoa_REPRODUCTION
AGP_PCOA_PUBLIC=results/AGP/four_generator_pcoa_REPRODUCTION
python3 -B experiments/AGP/four_generator_pcoa.py \
  --private "$AGP_PCOA_PRIVATE" --public "$AGP_PCOA_PUBLIC" --threads 4
python3 -B experiments/AGP/verify_four_generator_pcoa.py \
  --private "$AGP_PCOA_PRIVATE" --public "$AGP_PCOA_PUBLIC" --threads 2 \
  --output results/AGP/verification_four_generator_pcoa_REPRODUCTION.json
python3 -B experiments/AGP/four_generator_pcoa_cp.py \
  --private "$AGP_PCOA_PRIVATE" --ordination-public "$AGP_PCOA_PUBLIC" \
  --output results/AGP/four_generator_pcoa_cp_REPRODUCTION
```

To redraw CP categories directly from the original saved coordinates, omit
`--private` and `--ordination-public` from the final command, retaining a fresh
`--output`. The CP producer checks identities, prediction hashes and score-bin
counts; it does not refit coordinates or intervals. Each point inherits its
final-fit-0 score-cell interval, not an individual confidence interval.

## Class-level abundance panels (2026-10-22)

[four_generator_class_abundance.py](four_generator_class_abundance.py) renders
the five stacked-bar panels linked as the full supplementary figure in
[AGP_RDR.md](../../docs/AGP_RDR.md#class-level-relative-abundance).
It reuses the verified assessment inputs from `four_generator_pcoa.py`: all
2,002 real and 2,089 generated observations per method. The 614 feature-aligned
taxa are aggregated into 48 classes after float64 row closure. All classes,
including Unclassified, are retained. Colors follow the earlier class plots;
class order follows real mean abundance, and each source's samples are sorted
independently by their largest class abundance.

With the upstream studies available at the documented paths, use a fresh output:

```bash
MPLCONFIGDIR=/tmp/agp-class-abundance-mpl python3 -B experiments/AGP/four_generator_class_abundance.py \
  --output results/AGP/four_generator_class_abundance_REPLAY
```

Outputs are PNG/PDF figures, aggregate class means, closure/order diagnostics
and input/output hashes. The PDF retains vector text and rasterized bar layers.
Original arrays and studies are read without modification. The completed
[figure package](../../results/AGP/four_generator_class_abundance_20261022/manifest.json)
records the exact inputs and rendering source.

The compact main-text version uses the same observations and the original
48-class sorting, then displays the eight classes with the highest mean
abundance in the real assessment data and sums the other 40 as light-gray
Other. It uses a shared rank axis labeled `sample  index`, with independent
sample ordering within each source. The 6.8 × 3.6-inch PDF retains vector
text and axes; abundance regions are rasterized at 600 dpi. The full
48-class figure is unchanged.

```bash
MPLCONFIGDIR=/tmp/agp-class-abundance-mpl python3 -B experiments/AGP/four_generator_class_abundance.py \
  --compact --output results/AGP/four_generator_class_abundance_compact_REPLAY
```

The [compact package](../../results/AGP/four_generator_class_abundance_compact_20261026/manifest_compact.json)
includes PDF/PNG exports, aggregate class means, diagnostics, provenance, and
a [LaTeX figure and caption](../../results/AGP/four_generator_class_abundance_compact_20261026/figure_compact.tex).
It is a descriptive view of broad class-level patterns; visual similarity
alone does not establish distributional agreement.

## Generator and RDR fitting order

Full retraining is a separate, substantially larger reproduction. The
following stage map documents the included entrypoints; it is not a claim
that publication repeated these fits. Always use fresh roots and execute
each frozen source copy after its preparation stage.

| Component | Required stage order |
| --- | --- |
| Full MIDASim | `midasim_full.py prepare-generator --output NEW_ROOT`; frozen `midasim_full.slurm NEW_ROOT generate`; `midasim_full.py prepare-analysis --output NEW_ROOT` |
| Shared mask-conditional ICFM | `icfm_zero.py prepare --output NEW_ROOT --corrected /cwork/yx306/RDR/AGP/icfm_corrected_20261006`; frozen generator `train`, `generate`, `verify-generation` with `--output NEW_ROOT`; `icfm_zero.py prepare-analysis --output NEW_ROOT` |
| Mixture/NADE masks and conventional metrics | `mask_comparison.py initialize --output NEW_ROOT`; frozen source `fit-masks`, `generate`, `prepare`, `evaluate`, each with `--output NEW_ROOT`; independent `verify_mask_comparison.py --output NEW_ROOT` |
| Mixture/NADE RDR preparation | `mask_rdr.py initialize --output NEW_ROOT`; frozen source `generate --output NEW_ROOT --device cuda`, then `prepare --output NEW_ROOT` |

For each prepared P-versus-Q analysis, run 80 selection fits, freeze its own
selection, then run five final and five real–real control fits. The shared
runner accepts `selected_workflow.py fit --output ANALYSIS --stage selection
--task-index 0` (indices 0–79), `select --output ANALYSIS`, and `fit --output
ANALYSIS --stage final` / `--stage null` (indices 0–4). Complete the owning
wrapper's seal and independent verification stages before downstream export.
The MIDASim and ICFM launchers expose `fit selection`, `select`, `fit final`,
`fit null`, `seal-analysis`, and `verify`; require successful `afterok`
dependencies between stages when submitting Slurm jobs.

The original mask wrappers also carry an observed-mask resampling baseline;
their unchanged full-study completion gates cover that historical method.
It is excluded by `retained_mask_presentation.py` from the four-generator
paper output. The wrapper's full-study fit indices are 0–239 for selection
and 0–14 for final/null; these must not be confused with the per-pair indices
above. Its stage order is `initialize`, `generate`, `prepare`, selection
fits, `select`, final/null fits, `finish`, independent verification,
`mask_rdr_report.py report --out NEW_ROOT`, then `seal` and export.

Preparation wrappers retain pinned upstream paths. A new generator run does
not automatically redirect later scripts to its outputs. An end-to-end new
cohort/run requires consistent, recorded adaptations of those input paths
before freezing sources. The commands in the preceding section instead
reproduce the reported evaluation from the completed original fits.

For the paper presentation use `retained_mask_presentation.py` and
`four_generator_report.py`; the publication entrypoints omit obsolete
standalone generator-comparison report actions. Five-fit variation remains
conditional on fixed cohorts and generator fits. The previously viewed
assessment cohort makes this a retrospective evaluation, and neither public
hash checks nor private arithmetic replay establishes pointwise RDR accuracy.
