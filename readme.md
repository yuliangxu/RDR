# Relative Density Ratio (RDR)

Research code for *Distributional Evaluation of Generative Models via Relative
Density Ratio*. For densities p and q, the midpoint relative density ratio is
$r_0(x)=2p(x)/(p(x)+q(x))$ wherever $p(x)+q(x)>0$.

The simulation studies are complete, with reproducible code, a toy notebook,
results and computation costs. MNIST's retained results are selected and organized;
its DCGAN loss and sigmoid-slope selection study is complete. CelebA's expanded
selected-model assessment and five-family null study are complete, with
[paper results and figures](results/CelebA/README.md) retained in the repository.
Its selection grid records one numerical failure (319/320 fits completed).
AGP is still being selected and organized for the paper release.

## Experiment summaries

| Data | Summary | Code |
| --- | --- | --- |
| Simulations | [Simulations](docs/simulations.md) | `experiments/simulations/` |
| MNIST | [MNIST](docs/MNIST.md) | `experiments/MNIST/` |
| CelebA | [CelebA](docs/CelebA.md) | `experiments/CelebA/` |
| American Gut Project | [AGP](docs/AGP.md) | `experiments/AGP/` |

## Repository conventions

The intended final layout is:

```text
RDR/
├── utils/                    # Shared, dataset-independent RDR implementation
├── experiments/
│   ├── simulations/          # Toy notebook, model selection, and convergence
│   ├── MNIST/                # MNIST data, generators, and experiment workflows
│   ├── CelebA/               # CelebA data, generators, and experiment workflows
│   └── AGP/                  # Microbiome data, transforms, and workflows
├── docs/
│   ├── simulations.md
│   ├── MNIST.md
│   ├── CelebA.md
│   └── AGP.md
└── results/                  # Optional small figures/tables referenced by docs
```

`utils` owns reusable losses, neural architectures, fitting, plots, and
diagnostics. It must not depend on an experiment package, download datasets, or
change the working directory during import. Use a small number of coherent,
annotated modules; [the shared-code map](utils/README.md) describes the shared modules and dataset-specific ownership.

Each experiment folder owns its data preparation, population or generator,
configuration, entrypoints, and experiment-specific figures. Smaller experiments
can share these components within their data folder.

`docs` contains one summary per data group, with sections for its smaller
experiments: purpose, protocol, data and split roles, commands, outputs, results,
and limitations. Prefer this single narrative over duplicated final reports.

Create `results` when selected documentation needs small, necessary figures or
tables. Use matching data-group subfolders and relative links. Large datasets,
checkpoints, and raw predictions need acquisition or reconstruction instructions,
versions, and checksums. Local `/cwork` paths alone do not make a public release
reproducible.

## Reproducibility requirements for each retained experiment

1. Record input sources, preprocessing, generator provenance, and checksums;
   provide required split manifests and acquisition instructions.
2. State the RDR target, actual sampling denominator, P/Q counts, and roles of
   training, validation, calibration, and final evaluation. Record historical
   reuse and protocol influence.
3. Provide a tested environment and fixed configuration, including seeds,
   architecture, numerical conventions, and checkpoint-selection rule.
4. Document commands from a fresh checkout for preparation, fitting, evaluation,
   and reporting. Expose data/output paths and keep cluster launchers optional.
5. Distinguish saved-prediction replay, RDR refitting, and generator training.
   State which paths have been verified and the expected numerical agreement.
6. Link each retained table or figure to its generating command and inputs.
   Check completion counts before calling a multi-run experiment complete.

## Simulation release

The shared RDR implementation has been reorganized into five modules under
`utils`, with two small compatibility import modules. Dataset-specific helpers
now live in the four experiment folders, and callers and source snapshots have
been updated. The core migration preserves the existing scientific conventions.

Read [docs/simulations.md](docs/simulations.md) for the complete results and
[experiments/simulations/README.md](experiments/simulations/README.md) for
installation, a small execution check, full reproduction and optional Slurm
commands. All observations are synthetic; no data download is needed.

The retained studies comprise **1,150 model-selection fits**, **150 additional
matched-loss fits** (reusing 50 Hellinger references), and **12,000 convergence
fits**: **13,300 distinct scientific fits**. Convergence compares Hellinger, KL,
chi-square and JS at B8-Res64/output slope 2 over five settings, six training
sizes and 100 repetitions. A separate **1,200-fit timing replay** measures
training wall/CPU time and epochs without adding scientific repetitions.

At 5,000 observations per distribution, KL and JS have similar accuracy;
Hellinger performs worse under the common training recipe. Training averages
about four seconds per fit at n=1,000 and 12–18 seconds across settings/losses
at n=5,000 on one CPU thread. The [included evidence](results/simulations/README.md)
contains per-fit metrics, summaries, publication figures, protocols and
checksums. Full checkpoints are regenerated by the documented workflow.

Superseded simulation runners and duplicate results have been retired after
verified rollback backups. Frozen scientific sources, independent sample roles
and checkpoint rules are preserved; fit verification rejects incomplete grids
before reporting.

## Remaining datasets

MNIST's [reproducible workflows](experiments/MNIST/README.md) now cover VAE,
DCGAN, controlled digit perturbation and the real-versus-real validation null.
All four RDRs were refitted using JS with sigmoid slope 2, selected by the
completed 80-fit DCGAN Brier/local-calibration study. Public inputs are acquired
with recorded hashes; compact results and audits are in `results/MNIST/`.
Superseded MNIST scripts were retired after a verified full rollback archive.
CelebA's main workflow is in `experiments/CelebA/`, with 20 selected-model final
evaluations and 25 learned null fits/evaluations. Its repository package retains
the FID/model-selection/final/null tables and 32 current PNG/PDF figure pairs, including
compact score-ranked images with actual displayed ranges, per-bin supplements,
null RDR histograms, feature-attribute associations and response sensitivity, and C.1/C.2 diagnostics. The final calibration/evaluation halves are now merged
into one held-out test pool for all counts, intervals and neural summaries;
the earlier split-based outputs remain archived. Frozen sources and checksums
support archived-data replay; independent retraining still needs the external
inputs and path/environment work documented in the
[CelebA reproduction guide](results/CelebA/REPRODUCTION.md).
Unused historical CelebA entrypoints and superseded outputs were removed after
publishing the complete snapshot to
[RDR-working](https://github.com/yuliangxu/RDR-working/tree/celeba-working-20261003).
Four required CelebA support modules retain their `experiments/JRSSB/` paths.
AGP entrypoints retain their current locations.

The historical [requirements file](requirements.txt) contains conflicting pins,
including two PyYAML versions. Simulations have their own
[tested dependency versions](experiments/simulations/requirements.txt), and
MNIST has its own [environment recipe](experiments/MNIST/requirements.txt).
The remaining datasets still need environment recipes for the public release.

For each selected workflow, migrate dependencies, imports, launchers,
configuration paths, and archived-source lists together. Preserve objectives,
split provenance, and checkpoint compatibility; verify the reorganized workflow
from a fresh checkout before retiring its old paths.

The pre-cleanup snapshot remains in
[RDR-working](https://github.com/yuliangxu/RDR-working/tree/pre-reorganization-20260926),
tag `pre-reorganization-20260926` (commit `a6e19d7`).
