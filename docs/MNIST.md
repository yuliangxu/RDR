# MNIST experiments

This is the dataset-level inventory for the paper repository. **All experiment
families below are pending paper inclusion.** Listing an existing experiment
does not select it for the paper or certify a new reproduction run.

The planned code location is `experiments/MNIST/`. The links below describe the
current checkout; experiment entrypoints remain in place; dataset helpers have moved out of `utils`. Dataset loading,
pretrained generator adapters, and digit-specific plotting belong with these
experiments. Reusable RDR losses, ratio networks, and diagnostics belong in `utils/`.

## Experiment inventory

| Family | Purpose | Current entrypoint | Paper inclusion |
| --- | --- | --- | --- |
| Real versus VAE/DCGAN | Compare real images with frozen generators using pixel RDR. | [MNIST_generator_strict_split.py](../experiments/MNIST_generator_strict_split.py) | Pending |
| Controlled digit perturbation | Assess fitted RDR against known changes in digit sampling probabilities. | [MNIST_label_perturbation.py](../experiments/MNIST_label_perturbation.py) | Pending |
| Real-versus-real null | Inspect learned scores under a same-population comparison. | [MNIST_two_halves.py](../experiments/MNIST_two_halves.py) | Pending |
| Earlier interactive comparisons | Preserve exploratory generator comparisons for the selection review. | [MNIST_batch.py](../experiments/MNIST_batch.py), [notebook](../experiments/MNIST_batch.ipynb) | Pending |

The existing [MNIST result report](../experiments/JRSSB/MNIST_jrssb.md) records the
saved generator, perturbation, and null analyses, their provenance, and historical
reproduction evidence. Its figures and several linked inputs are outside Git.
This inventory does not replace or update those scientific results.

## Split roles and interpretation

The generator protocol reserves validation images from the official training
set and evaluates the official test set separately. Generated training and
validation examples are sampled online; they are not fixed Q datasets. Generator
pretraining is outside this RDR protocol and has separate provenance.

The perturbation protocol uses the official training set for fitting and divides
the official test set into disjoint validation and evaluation pools. Q resamples
images within each role, so repeated Q rows are not additional independent images.
Its digit-level reference also depends on the stated label-distribution assumptions.

The saved real-versus-real null uses the official test images for validation and
early stopping. It is a **validation diagnostic, without an independent final-test
null result**. A future independent-test null would require a new fit and protocol.

## Reproduction paths

The existing self-contained package is `/cwork/yx306/RDR/MNIST_jrssb_final/`.
Its `README.md` supplies the recorded commands for `workflow.py replay`,
`workflow.py train --experiment all --numerics historical`, and replay from newly
trained outputs. These are commands for that package, not a fresh GitHub clone.

The checkout contains the [workflow source](../experiments/JRSSB/mnist_final_workflow.py)
and the [recorded reproduction summary](../experiments/JRSSB/MNIST_jrssb.md#verified-end-to-end-reproduction).
Saved-evidence replay reconstructs tables and figures without refitting. Retraining
fits the RDR models using bundled data and frozen pretrained generators; it does
not reproduce generator pretraining. The package records a specific hardware and
software profile for exact numerical recovery. No reproduction was run for this inventory.

## Work needed for a portable paper release

- Select the experiment families before consolidating their entrypoints.
- Make raw-data and pretrained-generator acquisition, hashes, and configurable
  input/output roots explicit; the checkout currently uses local and HPC paths.
- Consolidate the relocated [MNIST helpers](../experiments/MNIST/helpers.py) within the MNIST experiment code,
  separating dataset/generator handling from shared plotting code.
- Replace interactive execution and fixed repository-path assumptions with explicit
  entrypoints where needed, while preserving split and input-scale conventions.
- Adapt package-building paths and report links after relocation. The existing
  package includes source snapshots and preserved image mosaics that are inputs
  to its replay; moving the checkout alone does not reproduce that package.
- Choose small figures/tables for optional `results/MNIST/`; keep larger assets
  separately with an acquisition guide and a manifest linking them to the report.
