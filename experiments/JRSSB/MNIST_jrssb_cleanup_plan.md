# MNIST final-result audit and cleanup plan

**Implemented after verification on 2026-09-13.**

The final workflow was verified twice against all 240,000 archived scores; generator weights and both PNGs also match exactly. The recovered auxiliary weights reproduce the archived scores. Fresh-extraction replay passed. Only after these checks, the 55 manifest-listed MNIST files were archived and removed from their original locations (29,841,942 bytes).

- Reproduction package: `/cwork/yx306/RDR/MNIST_jrssb_final/`.
- Exact evidence: `end_to_end_audit.json` in that package.
- Removal list and archive hash: `cleanup_receipt.json` in that package.
- Verified rollback archive: `/cwork/yx306/RDR/MNIST_retired_pre_final.zip`.
- The two README-referenced legacy entrypoints and external embedding cache remain on hold. Shared code, raw data and all unrelated experiments were untouched.

The sections below retain the **original pre-implementation audit and plan** for provenance. Their missing-model/reproduction limitations describe the starting state; the execution evidence above supersedes those limitations where recovery was demonstrated.

The authoritative scientific result is [MNIST_jrssb.md](MNIST_jrssb.md), including its pixel-space VAE/DCGAN comparison, controlled digit perturbation, and learned real-vs-real **validation** null diagnostic. Accepting this report as final does not convert the null into an independent-test experiment. Its stated limitations remain part of the final result.

Cleanup is limited to the explicit MNIST paths below. All new reproduction outputs and frozen artifact copies should go under `/cwork/yx306/RDR`, never into the existing repository result folders. This plan is documentation alongside the final report.

## Audit conclusion

Saved-score reconstruction of the final numbers and figure is supported by the available artifacts. Full historical training reproduction is not yet established. Preserve the saved evidence before removing superseded runs.

| Audit item | Finding and cleanup implication |
| --- | --- |
| Final figure integrity | All three input hashes and all four output hashes in `mnist-vae-dcgan-comparison/manifest.json` match the files currently present. |
| Generator score provenance | Both 10,000-row P and Q CSV groups match each branch's checkpoint scores to absolute tolerance 10^-14. All 480 panel-selection rows match the corresponding checkpoint scores at their recorded positions. |
| Hidden figure input | The renderer crops 12 image mosaics from `mnist-generator-trainval-testall/mnist_generator_strict_split_figure.png`. This older-looking PNG is indispensable to the current renderer and must stay. |
| Image replay limit | The final renderer retains the original mosaics, not independently regenerated test images. The generator checkpoints save scores, labels, weights and split indices, but not the generated test image tensors or latent tensors. Panel CSV agreement verifies score positions; it does not independently verify every raster thumbnail's identity. |
| Generator protocol | Saved role sizes are 55,000/5,000/10,000 real images, with disjoint training/validation indices. Inputs are pixels in [-1,1]. The old checkpoint configuration lacks explicit protocol, evaluation-scale and output-alpha fields; the current script has these switches. Preserve the original artifacts and record resolved settings rather than assuming current defaults define the historical run. |
| Split seed wording | Global seed is 42; the current paired generator split implementation uses `seed + 10`, i.e. 52. Frozen saved indices are authoritative. The report should distinguish these two seeds in its reproduction instructions. |
| Perturbation evidence | Bundle contains train P/Q 60,000/60,000 and test P/Q 5,000/5,000 scores, label probabilities and sampled indices; validation membership is saved. It has no fitted model state or complete loss history. Preserve scores as the final numerical evidence. |
| Null evidence | Bundle contains train P/Q 30,000/30,000 and validation P/Q 5,000/5,000 scores. It has no fitted model state or complete loss history, and no independent-test result. A new three-role null would be a new experiment, not a reproduction of this final table. |
| Training source provenance | `utils/DRE_batch.py` and `utils/DRE_func.py` are modified in the working tree and are shared with CelebA/AGP. A current source snapshot is not proof of the exact historical training implementation. Do not reset, prune or overwrite these utilities. |
| Environment | Root `requirements.txt` is repository-wide, includes two conflicting PyYAML pins, and is not a verified historical MNIST environment lock. Create a separate tested MNIST environment specification; leave the shared file untouched. |
| Data duplication | All eight files in `data/MNIST/raw` match the external `MNIST/MNIST/raw` copy byte-for-byte. Both paths are currently used by retained scripts, so neither copy is a deletion candidate yet. |
| Logs | Jobs 53679306 and 53679669 both report the final paired result and write to the same output directory. Keep both logs; their agreement alone does not establish which job produced each surviving byte. |
| ZIP | `/cwork/yx306/RDR/MNIST_jrssb.zip`, previously delivered in this conversation, is absent at this audit. Do not count it as a backup or assume it can replace source artifacts. Rebuild a verified package later. |

## Keep: final result and its dependency closure

Keep these complete directories initially; their combined saved-result size is about 23.4 MB. Removing small derived CSVs now offers little benefit and can break report links.

| Exact location | Purpose | Files / bytes |
| --- | --- | ---: |
| `/cwork/yx306/RDR/mnist-generator-trainval-testall/` | Both ratio checkpoints, test scores/summary, split manifest, original PNG required by renderer, original PDF | 7 / 10,948,528 |
| `/cwork/yx306/RDR/mnist-vae-dcgan-comparison/` | Final PNG/PDF, plotting script, panel-selection CSV and manifest | 5 / 2,067,400 |
| `experiments/results/MNIST_label_perturbation/` | Final perturbation score/split bundle and all eight accompanying tables/artifacts | 8 / 7,139,692 |
| `experiments/results/MNIST_two_halves/` | Final learned validation-null bundle, scores and summary | 3 / 3,258,171 |

Retain these repository files:

- `experiments/JRSSB/MNIST_jrssb.md` and this cleanup plan.
- `experiments/MNIST_generator_strict_split.py` and `experiments/MNIST_generator_strict_split.slurm`.
- `experiments/MNIST_label_perturbation.py` and `experiments/MNIST_two_halves.py`.
- `utils/MNIST_help.py` and the existing package initializer files needed for imports.
- `utils/DRE_batch.py` and `utils/DRE_func.py` as protected shared dependencies, with no edits or deletion under this cleanup.

Retain `/cwork/yx306/RDR/out/mnist-generator-strict-53679306.out`, its `.err` companion, and the corresponding `.out`/`.err` files for job 53679669.

Retain these external training/scoring inputs in place:

- `/hpc/group/mastatlab/yx306/MNIST/mnist_vae/vae.py` and its package directory, including existing package support files.
- `/hpc/group/mastatlab/yx306/MNIST/mnist_vae/vae_epoch_25.pth`.
- `/hpc/group/mastatlab/yx306/MNIST/mnist_dcgan/netG_epoch_99.pth`.
- `/hpc/group/mastatlab/yx306/MNIST/MNIST/raw/` and repository `data/MNIST/raw/` until retained entrypoints use one explicitly configured location.

External generator SHA256 fingerprints:

| File | SHA256 |
| --- | --- |
| `vae.py` | `5216a92fe21b73698538eb1f48ee1fe20624eb17f64f0120891b23f261e8c34c` |
| `vae_epoch_25.pth` | `233916a8095759cf732f4dc19c381bc8de03d97040e68f4252a257a5502a9c70` |
| `netG_epoch_99.pth` | `d61db3ef26f469f29c1725b16c2be82cecd1a53fc53bcdda9f8789595a3d65f3` |

## Candidate removals after reproduction verification

The following five exact output directories are not inputs to the final report or renderer. Total: **29,648,190 bytes (29.6 MB)**. They are candidates for removal from the active workspace after the keep set has been frozen and checked. They have not been deleted or declared redundant copies of the final result.

| Directory under `/cwork/yx306/RDR/` | Reason excluded from final result | Bytes |
| --- | --- | ---: |
| `mnist-generator-strict-split/` | Earlier 60,000 train / 5,000 validation / 5,000 test protocol | 9,966,972 |
| `mnist-generator-legacy-testval-alpha1-dcgan/` | Official test data reused for validation; different DCGAN configuration | 6,100,556 |
| `mnist-generator-legacy-testval-alpha1-dcgan-rescore-eval01/` | Rescoring of that legacy run in [0,1] | 1,768,550 |
| `mnist-generator-trainval-testall-alpha1-dcgan-eval01/` | DCGAN-only [0,1] evaluation variant | 5,894,826 |
| `mnist-generator-trainval-testall-alpha2-dcgan/` | Different output-alpha DCGAN-only run | 5,917,286 |

Associated source candidates, all under `experiments/`:

| Exact files | Reason / condition |
| --- | --- |
| `MNIST_generator_alpha1_dcgan_eval01.slurm`, `MNIST_generator_alpha1_dcgan_legacy.slurm`, `MNIST_generator_alpha2_dcgan.slurm` | Launchers for excluded variants. |
| `MNIST_generator_rescore_eval_scale.py` | Alternate evaluation-scale utility, not needed by the final renderer or paired training path. |
| `MNIST_inception_embeddings.py`, `MNIST_two_halves_inception_embeddings.py` | Feature-space experiments outside the final pixel result. |
| `MNIST_generator_fid_features.py`, `MNIST_generator_fid_features.slurm`, `MNIST_generator_fid_features.md` | Separate feature/FID follow-up, not a source for the final report. The report's referenced `experiments/results/MNIST_generator_fid_features_55325692/` directory is absent in the audited checkout; do not invent a deletion target or claim its results were recovered. |
| `MNIST_batch.py`, `MNIST_batch.ipynb` | Earlier interactive experiment, not a final dependency. **Hold for reference resolution:** root `readme.md` names both. Leave that unrelated shared document untouched in this audit; do not silently create broken references during cleanup. |

After their run directories are retired, the exact `.out`/`.err` pairs below are candidate removals under `/cwork/yx306/RDR/out/`:

- `mnist-generator-strict-53679140`.
- `mnist-dcgan-alpha1-legacy-53687824`.
- `mnist-dcgan-alpha1-eval01-53689365`.
- `mnist-dcgan-alpha2-53687229`.

Bytecode files may be removed only by enumerating exact MNIST module filenames in `experiments/__pycache__/` and `utils/__pycache__/MNIST_help.*.pyc`. Never remove an entire shared `__pycache__` directory or unrelated bytecode. Recheck the file list immediately before execution.

The 1,720,888,848-byte `/hpc/group/mastatlab/yx306/MNIST/inception_embeddings/` cache is unnecessary for this pixel report, but lives in shared external storage and has not been audited for consumers outside this checkout. **Defer removal**; it is not included in the 29.6 MB removal set. No changes to external model or data directories are proposed now.

## Ordered implementation plan

1. **Freeze the final evidence.** Create `/cwork/yx306/RDR/MNIST_jrssb_final/` with copies of the report, retained results, original source PNG, figure renderer, MNIST scripts, exact shared-source snapshots and generator provenance. Record relative paths, SHA256 hashes, original locations and the current Git state. Keep the working shared utilities untouched. Treat copied current code as an audit-time snapshot unless historical provenance can be recovered.

2. **Add a saved-result reproduction entrypoint.** It should regenerate all report tables, target-adjusted digit errors, null RMSEs and both figure formats from the frozen scores, checkpoints and PNG. Replace the renderer's absolute source default with an explicit package input path. Retain the original raster panels unless independently verified original images/latents are recovered. Store all newly produced outputs under `/cwork/yx306/RDR/MNIST_jrssb_final/`; do not overwrite authoritative inputs.

3. **Document training separately from saved-result reconstruction.** Pin `trainval_testall`, seed 42 / split seed 52, batch size 512, validation size 5,000, VAE budget 3 epochs, DCGAN budget 20 epochs, output alpha 0.5, patience 5, ten validation batches, and model-space [-1,1] evaluation for the generator run. Pin seed 123 and the existing role assignments for perturbation/null. Snapshot all sampler/loss/optimizer defaults. Record that exact retraining of the perturbation/null models is unverified and their weights are missing; a new fit cannot be presented as recovery of those weights. Do not replace the final validation-null result with a new independent-test experiment as part of cleanup.

4. **Make the package verifiable.** Add a tested MNIST-only environment specification and explicit commands for table/figure replay. Record Python/library versions, device and relevant numerical settings; these describe the replay environment, not necessarily the historical training environment. Include the missing dependencies in the reproduction package: both ratio checkpoints, the original source PNG, shared code snapshots, generator implementation/weights or verified retrieval instructions, and data fingerprints. A report-and-figures ZIP alone is not the training reproduction package.

5. **Validate before removal.** Reconstruct from a fresh extraction using only the keep set. Check every report link; counts and disjoint role indices; checkpoint/CSV agreement; all table values at reported precision; panel score membership; identical histogram bins and digit groups; and valid PNG/PDF rendering. Compare numerical content and image panels rather than requiring byte-identical PDFs across environments. Any training smoke checks must write to a new directory and must not replace final results. Verify all package hashes and ZIP integrity.

6. **Execute only the reviewed MNIST file list.** Recheck Git status, symlink targets, references, and active jobs; stop on any changed or ambiguous candidate. Produce a path-by-path dry-run manifest and only then remove the listed superseded MNIST files. Do not use broad `MNIST*`/`mnist*` recursive deletion, delete `/cwork/yx306/RDR/out/`, reset shared utilities, or clean the whole repository. Decide explicitly how to handle the two README-referenced legacy entrypoints before including them. Remove frozen historical archives as well only if they are no longer needed for the agreed retention policy.

7. **Revalidate and deliver.** Run the same saved-result reconstruction and link checks after cleanup. Rebuild `MNIST_jrssb.zip` from the verified final package, and record exactly what was removed. Keep the scientific values and limitations in `MNIST_jrssb.md` unchanged apart from reproducibility clarifications and relocated links.

## Protected scope

No deletion or editing of CelebA, AGP, UOT, simulation/benchmark experiments, `agent3.md`, other JRSSB agents, their outputs, unrelated logs, shared utilities, repository-wide requirements, or root documentation is included. The `agent3.md` link is background context and does not make CelebA a cleanup dependency. Do not recursively copy or prune the whole `JRSSB`, `data`, `utils`, or `experiments/results` directory.

The initial planning phase created only this plan. Implementation and verification outcomes are recorded at the top of this file and in the execution audit and cleanup receipt.
