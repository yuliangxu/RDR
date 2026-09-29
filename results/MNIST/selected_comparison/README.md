# Refit comparison using JS and sigmoid slope 2

[Figure PNG](mnist_vae_dcgan_comparison.png),
[figure PDF](mnist_vae_dcgan_comparison.pdf), and [report](RESULTS.md).

Both RDR networks were trained afresh using the frozen DCGAN-selected
configuration; VAE received that configuration without another search. The
pretrained VAE/DCGAN generators remained fixed. This is a new result alongside
the historical Hellinger comparison in the parent directory.

| Generator | Validation-selected epoch | Whole-test Brier | Local absolute calibration gap |
| --- | ---: | ---: | ---: |
| VAE | 19 | 0.000748 | 0.001505 |
| DCGAN | 17 | 0.082477 | 0.018135 |

The real fitting/validation/test counts are 55,000/5,000/10,000, exactly matching
the historical split. The whole-test Brier uses 10,000 P and 10,000 Q. Local
discrepancy uses disjoint 5,000-P/5,000-Q calibration and evaluation roles.
VAE supported evaluation mass is 0.9994; DCGAN supported mass is 1.0.

VAE's middle region [0.7,1.2) contains only 0.05% of evaluation mixture mass;
40% of that mass has calibration support, and its supported gap is 0.5283.
For DCGAN, the region has 8.1% mass, full support and gap 0.0455. Small aggregate
gaps therefore do not establish accuracy in sparse regions or for individuals.

These files are byte-identical copies from
`/cwork/yx306/RDR/MNIST/selected_comparison_20260928/`, recorded in
[manifest.json](manifest.json). Slurm job 56703307 completed successfully in
3 minutes 19 seconds. The [independent audit](verification.json) checks the
split, locks, saved metrics, calibration cells, hashes, and all 480 panel rows.

The full run contains frozen source/data/weights, split indices, both RDR
checkpoints, and the newly saved evaluation images and scores. This smaller
repository evidence folder alone is not a standalone training/replay package.
Follow the [workflow instructions](../../../experiments/MNIST/README.md#comparison-refit-with-the-selected-configuration)
to retrain or replay from the full run.

Only Brier and local calibration diagnostics assess the new fits. Differences
from historical results also reflect the changed checkpoint rule and VAE
training schedule. Test images were inspected previously, and selected image
panels do not estimate prevalence. See the report for exact scope and CI meaning.
