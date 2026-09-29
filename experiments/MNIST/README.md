# Reproducible MNIST workflows

[Results and statistical scope](../../docs/MNIST.md) cover real versus VAE,
real versus DCGAN, controlled digit perturbation and the real-versus-real
validation null. All four selected-model fits use JS and `r=2 sigmoid(2z)`.
The DCGAN search is retained so the chosen configuration can also be reproduced.

## Environment and public inputs

Use Python 3.9 or a compatible environment. The recorded GPU environment is
Python 3.9.25, PyTorch 2.6.0+cu124, torchvision 0.21.0, NumPy 1.26.4,
SciPy 1.13.1 and matplotlib 3.9.4 on an RTX A5000. CPU smoke runs are supported;
exact floating-point agreement across different hardware is not promised.
Use this experiment's dependency file rather than the historical root-wide pins:

```bash
python3 -m pip install -r experiments/MNIST/requirements.txt
python3 -B experiments/MNIST/inputs.py --data-root data
```

The acquisition command downloads four raw MNIST files from the public
PyTorch MNIST mirror and the two pretrained weights from
[csinva/gan-vae-pretrained-pytorch](https://github.com/csinva/gan-vae-pretrained-pytorch).
All six downloads were tested and match the original experiment SHA-256 hashes.
Files are verified before installation; mismatched existing files are preserved
and rejected. Check an existing input tree without downloads:

```bash
python3 -B experiments/MNIST/inputs.py --data-root /path/to/inputs --verify-only
```

The layout is:

```text
inputs/
  MNIST/raw/{train,t10k}-{images-idx3,labels-idx1}-ubyte
  mnist_dcgan/netG_epoch_99.pth
  mnist_vae/vae_epoch_25.pth
```

`generators.py` contains only the matching generator architectures/loaders.
No external VAE Python script, private dataset path or old result folder is
needed. Frozen generator pretraining is not rerun; its training membership
has not been independently audited.

## Refit the four retained contrasts

Run commands from the repository root. Preparation rejects an existing output
directory, verifies the published selection evidence and freezes all required
source and inputs. Subsequent commands use that snapshot:

```bash
python3 -B experiments/MNIST/comparison.py prepare \
  --output runs/mnist-comparison --data-root data
python3 -B runs/mnist-comparison/source/experiments/MNIST/comparison.py run \
  --output runs/mnist-comparison --device cuda

python3 -B experiments/MNIST/controls.py prepare \
  --output runs/mnist-controls --data-root data
python3 -B runs/mnist-controls/source/experiments/MNIST/controls.py run \
  --output runs/mnist-controls --device cuda
```

For CPU end-to-end checks, add `--smoke` to each prepare command, use separate
output directories, and use `--device cpu` during the run. Smoke outputs are
explicitly marked; they are not scientific results. `train --generator vae`
(or `dcgan`) and `train --experiment perturbation` (or `null`) run one branch.
Completed branches are hash-verified and reused, allowing interrupted runs
to resume; incomplete branches are retrained deterministically.

On the recorded HPC cluster, the Slurm wrappers run both branches and render:

```bash
sbatch --output=runs/mnist-comparison/logs/%j.out \
  --error=runs/mnist-comparison/logs/%j.err \
  experiments/MNIST/comparison.slurm "$PWD/runs/mnist-comparison"
sbatch --output=runs/mnist-controls/logs/%j.out \
  --error=runs/mnist-controls/logs/%j.err \
  experiments/MNIST/controls.slurm "$PWD/runs/mnist-controls"
```

Adjust account/partition directives for another cluster. The direct Python
commands work without Slurm. `RDR_PYTHON` can select another interpreter for
the wrappers. They clear `PYTHONPATH` and use the frozen source tree.

## Training and calibration protocol

All refits use the selected base-64 CNN, batch size 512, Adam, OneCycle maximum
learning rate 0.0006, gradient norm cap 1, maximum 20 epochs, minimum stopping
epoch 6, patience 5, and BatchNorm frozen from zero-based epoch 5. P/Q share a
single balanced forward pass. The JS objective is balanced classification
cross-entropy for probability r/2, minus log(2), computed stably from logits.
Checkpoints minimize fixed validation Brier and are saved/restored before new
assessment or diagnostic scores are accessed. Every training row is consumed.

Comparison reconstructs the exact 55,000/5,000/10,000 historical real split,
with fresh generated training draws and fixed independent validation/test Q.
Control primary roles exactly preserve historical rows: perturbation has
60,000/60,000 fitting, 5,000/5,000 validation, 5,000/5,000 held-out assessment;
null has disjoint 30,000/30,000 fitting and 5,000/5,000 validation only.
Q perturbation resamples within the role's pool at the documented digit weights.

Generator calibration uses disjoint halves of the official-test score pool.
Control calibration uses four fresh independent 5,000-draw streams from fixed
empirical references after checkpoint freezing: perturbation's held-out 5,000
images, or null's pooled 10,000 validation images. Thus control CIs describe
conditional finite-pool resampling uncertainty and exclude reference-image
sampling uncertainty. Null has no independent final test. All intervals concern
cell-average RDR, not individual-image guarantees. See [MNIST.md](../../docs/MNIST.md)
for counts, middle-region support, retrospective-test scope and exact results.

## Replay saved results

After a run, regenerate its figures/tables without fitting:

```bash
python3 -B runs/mnist-comparison/source/experiments/MNIST/comparison.py report \
  --output runs/mnist-comparison
python3 -B runs/mnist-controls/source/experiments/MNIST/controls.py report \
  --output runs/mnist-controls
```

Reports verify all frozen inputs and complete-fit hashes. A root `COMPLETE`
marker is written only after both branches and report/figure generation finish.
Comparison saves full new image tensors and scores, exact panel positions,
checkpoints, histories and calibration cells. Controls save all primary and
fresh diagnostic scores, source indices, labels, checkpoints and digit tables.
Source-hash checks remain strict: replay older runs with their own saved source.
The compact `results/MNIST/` folders alone do not contain full replay inputs.

## Reproduce DCGAN configuration selection

The selected configuration is verified by `selection_evidence.py` from the
published protocol/selection locks and all 80 per-fit rows. A fresh refit does
not need to repeat the search. To reproduce the search itself:

```bash
python3 -B experiments/MNIST/model_selection.py prepare \
  --output runs/mnist-selection --data-root data
for task in $(seq 0 79); do
  python3 -B runs/mnist-selection/source/experiments/MNIST/model_selection.py fit   \
  --output runs/mnist-selection --device cuda --task-index "$task"
done
python3 -B runs/mnist-selection/source/experiments/MNIST/model_selection.py freeze \
  --output runs/mnist-selection
for repeat in 0 1 2 3 4; do
  python3 -B runs/mnist-selection/source/experiments/MNIST/model_selection.py evaluate   \
  --output runs/mnist-selection --device cuda --repeat "$repeat"
done
python3 -B runs/mnist-selection/source/experiments/MNIST/model_selection.py report \
  --output runs/mnist-selection
```

For a cluster array, use `selection.slurm` with array 0-79 for `fit`, then its
`finalize` stage with an `afterok` dependency on the whole array. Configuration
is in `selection_config.json`. The 16 Hellinger/KL/chi-square/JS × sigmoid-slope
candidates share initialization, data order and generator streams within each
of five repeats. Brier chooses checkpoints; a paired one-SE Brier shortlist
then minimizes local gap with at least 99% supported mass. No other risk metric
selects a configuration. The expanded midpoint losses all target 2p/(p+q).

## Tests and source layout

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python3 -B -m pytest -q -p no:cacheprovider tests/test_mnist_*.py tests/test_calibration.py
```

Two optional original-generator parity tests use `MNIST_PRETRAINED_ROOT` and
`MNIST_LEGACY_HELPERS`; they skip when those archival inputs are unavailable.
They passed against the original pretrained checkpoints during migration.
Other tests run without private HPC files. Public input downloads, preparation
and both CPU smoke workflows were checked with clean exported code.

Active files are the three runners and Slurm wrappers; public input acquisition;
minimal generator adapters; compact-evidence verification; selection metrics;
and the selection/comparison/control renderers. Shared neural networks and
calibration methods remain under `utils/`. Superseded interactive scripts,
wrappers and packaging tools are preserved in the verified rollback archive,
recorded in [the cleanup receipt](../../results/MNIST/cleanup_20260929.json).
