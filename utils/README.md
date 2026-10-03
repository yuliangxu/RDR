# Shared RDR code

`utils` contains dataset-independent RDR methods. Dataset loading, generator
adapters, population construction, and dataset-specific image or microbiome
plots live under `experiments/`. Shared modules do not import those packages,
download data, or change the working directory.

## Module guide

| Module | Responsibility |
| --- | --- |
| [losses.py](losses.py) | Model- and score-based objectives, with explicit denominator, output-scale, clipping, and normalization conventions. |
| [networks.py](networks.py) | Bounded heads, MLPs, residual blocks, tensor-only image networks, and initialization. Historical class/state-dictionary names are preserved. |
| [training.py](training.py) | Fitting, validation, checkpoint restoration, prediction-related helpers, and generic loader mixtures. Distinct historical fitting recipes remain explicit. |
| [diagnostics.py](diagnostics.py) | Evaluation, ratio summaries, classifier baseline, extreme selection, and reusable plots. Plotting/pandas/scikit-learn imports are deferred until needed. |
| [calibration.py](calibration.py) | Fixed and adaptively merged score-cell intervals and true within-cell variance bounds. |

Two small modules, [DRE_func.py](DRE_func.py) and [DRE_batch.py](DRE_batch.py),
re-export shared implementations for existing callers and archived class names.
They contain no separate fitting implementation or dataset-specific imports.
New code should import from the five modules above. The former catch-all
`help_func.py` has been removed.

## Loss targets and compatibility

Let R denote the distribution actually supplied as the denominator. A function
argument named `x_q` does not prove that those samples came from Q: historical
callers also pass samples from M = (P + Q) / 2. Each function documents its actual
objective, input scale, and separate sample means.

| Objective | Unrestricted population target / convention |
| --- | --- |
| `Hellinger_loss` | For 0 < xi < 1, `(xi / (1-xi)) * p/r`; xi=0.5 gives p/r. Use midpoint denominator samples for RDR. |
| `KL_loss`, `Chisq_loss`, `JS_loss` | Target p/r within the model's supported range. Historical additive constants and forward-pass conventions are retained; JS requires outputs in (0,2). |
| `hellinger_stable` | Historical score-input Hellinger objective. Despite its name, its optional log/clipping arguments were inactive and remain so. |
| `kl_from_outputs` | Log-ratio inputs, with its original additive-constant convention. |
| `chisq_from_outputs` | Symmetric squared objective targets p/(p+r), unlike `Chisq_loss`. It is not interchangeable with the density-ratio objective. |
| `original_hellinger_midpoint_loss` | P and Q forwarded together; scores clipped before separate empirical means. |
| `midpoint_hellinger_loss` | P and Q forwarded separately; no output clipping. |
| `concatenated_midpoint_hellinger_loss` | P and Q forwarded together; no output clipping. |

All three explicit midpoint losses preserve the weights
`0.5 E_P[r^-1/2] + 0.25 E_P[sqrt(r)] + 0.25 E_Q[sqrt(r)] - 1`.
Separate means retain equal P/Q mixture weights even when sample sizes differ.
Mixed versus separate forwards can differ with BatchNorm, so these variants
remain distinct. Experiments choose the variant matching their recorded protocol.

The new simulation protocol uses `midpoint_rdr_loss` for the four explicitly
expanded midpoint objectives, `make_ratio_mlp` for the common bounded output
and five architectures, and `fit_ratio_mlp` for controlled fitting with absolute
best-validation checkpoint restoration. These are shared, dataset-independent
APIs; historical entrypoints retain their recorded conventions.

Training entrypoints retain optimizer, initialization, scheduler, and stopping
behavior. In particular, older full-batch early stopping and newer experiment
recipes do not share one checkpoint-selection rule. Some pre-existing optional
legacy model branches remain unavailable, as documented in `training.py`; this
refactor does not add implementations for them.

Calibration intervals target population **cell-average RDR**, not individual RDR
values. A caller must supply the appropriate independent calibration data and
respect the partition-selection assumptions. A whole-space interval [1,1] does
not establish pointwise equality to one or zero within-cell variance.
`plot_calibration_cells` plots count-based estimates and intervals against
neural cell means, with optional support-based marker sizes. It needs no oracle
densities and omits empty-cell estimate placeholders.

## Dataset-owned code

| Destination | Contents moved out of shared utilities |
| --- | --- |
| [MNIST workflows](../experiments/MNIST/README.md), [generator adapters](../experiments/MNIST/generators.py), [controls](../experiments/MNIST/controls.py) | Public digit inputs, pretrained generators, controlled digit sampling, null validation and reproducible figures. |
| [CelebA workflows](../experiments/CelebA/README.md) | Expanded generator pools, selection, merged-test diagnostics, null controls and attribute analysis. |
| [AGP helpers](../experiments/AGP/helpers.py), [microbiome plots](../experiments/AGP/microbiome.py) | Composition transforms, phylogenetic handling, and microbiome-specific diagnostics. |
| [Simulation populations](../experiments/simulations/population.py), [toy notebook](../experiments/simulations/toy_illustrations.ipynb) | Gaussian-mixture populations, analytic truth, noisy lifting, and Gaussian/Beta illustrations. |

The simulation workflow is organized around model selection and convergence;
the remaining datasets are being selected and reorganized separately.

## Verification and frozen sources

The reorganization was compared against the pre-cleanup code for loss values and
gradients, seeded networks and state dictionaries, tiny training histories and
restored states, seeded samplers, summaries, and rendered plots. Existing CI,
architecture, AGP, and CelebA tests also run against the shared modules.

From the repository root:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLBACKEND=Agg \
  python3 -B -m pytest -q -p no:cacheprovider tests \
  experiments/AGP/test_*.py
```

New source snapshots include the extracted dependency closure. Historical source
hash checks remain strict: replay an old run with its frozen source tree or the
`pre-reorganization-20260926` checkout, not by bypassing provenance validation.
MNIST's historical audit is retained in its frozen reproduction package;
the current [MNIST workflow](../experiments/MNIST/README.md) provides portable
input acquisition, selected-model refitting and replay from new frozen runs.
