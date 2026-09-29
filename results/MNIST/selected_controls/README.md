# Selected-model digit perturbation and validation null

[Results](RESULTS.md) · [main PNG](mnist_selected_controls.png) ·
[main PDF](mnist_selected_controls.pdf) ·
[calibration PNG](mnist_controls_calibration.png) ·
[calibration PDF](mnist_controls_calibration.pdf).

Both RDRs were trained afresh using DCGAN-selected JS loss and sigmoid slope 2.
Perturbation selected epoch 16 and has held-out Brier 0.193262. Null selected
epoch 8 and has **validation** Brier 0.253278; it has no independent final test
and is slightly worse than constant-r=1 Brier 0.25.

Conditional finite-pool local gaps are 0.055599 and 0.091185 respectively.
The report explains independent resampling, reference-pool uncertainty,
middle-region mass/gaps and exact primary counts. Primary index sequences
reproduce the historical experiments exactly.

Files in [manifest.json](manifest.json) are exact copies from
`/cwork/yx306/RDR/MNIST/selected_controls_20260929/`, produced by job 56707707.
The [independent audit](verification.json) verifies 210,000 primary scores,
40,000 diagnostic scores, all 40 calibration cells, checkpoint selection,
indices, hashes and representative CPU checkpoint predictions. The
[protocol audit](protocol_verification.json) checks historical row identity.

This is compact evidence. Full checkpoints, score/index tensors, inputs,
splits and frozen source remain in the local run. The
[portable workflow](../../../experiments/MNIST/README.md) recreates these
inputs and fits from a fresh clone, then replays reports from the saved run.
