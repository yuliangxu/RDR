# Selected MNIST evidence

Current JS/slope-2 fits: [VAE/DCGAN](selected_comparison/README.md) and
[digit perturbation / validation null](selected_controls/README.md).
The [MNIST summary](../../docs/MNIST.md) lists all current results and
reproduction commands. The historical evidence below remains preserved.

These are byte-identical copies of the retained historical evidence. Real versus
VAE, real versus DCGAN, and controlled digit perturbation are the three main
results. The real-versus-real null is retained as a **validation diagnostic**;
it has no independent final-test result.

| Evidence | File | Role |
| --- | --- | --- |
| Combined VAE/DCGAN figure | [PNG](mnist_vae_dcgan_comparison.png), [PDF](mnist_vae_dcgan_comparison.pdf) | Main results 1–2 |
| Generator comparison table | [generator_table.md](generator_table.md) | Main results 1–2 |
| Controlled digit table | [digits_table.md](digits_table.md) | Main result 3 |
| Learned null table | [null_table.md](null_table.md) | Validation diagnostic 4 |
| Historical numeric summaries | [metrics.json](metrics.json) | All four retained results |

[manifest.json](manifest.json) records each original absolute source path, byte
count, and SHA256. The PNG/PDF come from the original combined comparison folder;
the tables and metrics come from the existing frozen package's saved replay.
Copying this evidence does not constitute new fitting, new scientific replay, or
an independent confirmation of the historical results. The new DCGAN model
selection study is separate and is not included in these historical summaries.

Full frozen data, scores, models, source snapshots, and historical audits remain
under `/cwork/yx306/RDR/MNIST_jrssb_final/`; the original combined figure remains
under `/cwork/yx306/RDR/mnist-vae-dcgan-comparison/`. Raw datasets, original result
folders, and frozen archives were left in place. The incompatible older
perturbation validation table was not copied here.

See the [MNIST summary](../../docs/MNIST.md) for split accounting, interpretation,
limitations, and current code, and the
[historical report](../../experiments/JRSSB/MNIST_jrssb.md) for detailed provenance.

The separate [DCGAN model selection evidence](model_selection/README.md) contains
the newly completed 80-fit loss/sigmoid comparison and five final assessments.
It preserves its own protocol, selection lock, tables, figures and manifest.

The [new VAE/DCGAN comparison](selected_comparison/README.md) retrains both
RDR networks using the selected JS loss and sigmoid slope 2. Its new
[PNG](selected_comparison/mnist_vae_dcgan_comparison.png) and
[PDF](selected_comparison/mnist_vae_dcgan_comparison.pdf) are separate from
the retained historical Hellinger figure above.

The [reproduction verification](reproduction_verification.json) records the
public input acquisition, clean-export tests and portable selected-model smoke
runs after the final MNIST cleanup.

[Relocation verification](relocation_verification.json) records scientific AST
preservation against the original tracked code, a fresh saved-evidence replay,
and the 33 passing target/calibration/workflow tests. This is separate from
the historical GPU retraining audits bundled under `/cwork`.

The later [cleanup receipt](cleanup_20260929.json) records 21 retired legacy
source files and their verified full rollback archive. Active code now contains
the selected workflows, input acquisition and the configuration-selection
procedure; all historical numeric evidence and raw inputs remain preserved.
