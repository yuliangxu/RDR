# Retained research support

The finalized CelebA pipeline and commands are in
[experiments/CelebA](../CelebA/README.md); its results are summarized in
[docs/CelebA.md](../../docs/CelebA.md).

Four modules remain here because the active expanded workflow uses them:

- `CELEBA_agent3.py`: pinned generator loading, preprocessing and atomic output.
- `CELEBA_data.py`: input manifests, transforms, hashing and seeds.
- `CELEBA_inception.py`: canonical FID Inception features and provenance.
- `CELEBA_rdr.py`: the recorded midpoint Hellinger estimator used in null diagnostics.

`configs/agent3.yaml` retains the generator/seed provenance read by expanded
pool preparation. Historical counts in this input are superseded by the
expanded configuration. The requirements files record earlier environments;
see the [current reproduction scope](../../results/CelebA/REPRODUCTION.md).

Unused smaller-data workflows have been archived in
[RDR-working](https://github.com/yuliangxu/RDR-working/tree/celeba-working-20261003).
Exact historical replay uses the corresponding frozen sources and data.
The [CI specification](CI/agent_CI.md), compatibility calibration import,
and MNIST reference document remain for the other experiments.
