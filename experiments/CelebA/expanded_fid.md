# Expanded-pool FID check

This check uses the approved fixed Diffusion-StyleGAN2 checkpoint, with
truncation 0.650 for Q_L and 1.569 for Q_U. It generates additional samples
and computes FID; it does not fit an RDR model or retune either generator
distribution. The existing model-selection jobs continue independently.

| Pool | P images | Images per Q branch |
| --- | ---: | ---: |
| Training | 122,984 | 120,000 |
| Validation | 39,786 | 40,000 |
| Final calibration/evaluation | 39,829 | 40,000 |

P uses the complete archived train and validation pools. Final P combines
19,867 historical design images with 19,962 historical test images. Within
each pool, both Q comparisons use exactly the same P reference. This is a
retrospective comparison because the design/test images and generator
settings have previously been inspected.

Each Q training pool already contains 120,000 images. Each validation pool
adds 20,000 new images to its existing 20,000. Each final pool combines the
existing 10,000 design and 10,000 test images with 20,000 new images. The
80,000 new samples use disjoint seed intervals starting at 20,000,000
(validation lower), 21,000,000 (validation upper), 22,000,000 (final lower),
and 23,000,000 (final upper). Preparation checks overlap against every
historical CSV manifest, then freezes these choices before computing FID.

## Results and execution

The durable run is
`/cwork/yx306/RDR/CelebA/expanded_fid_20260929`.
Its [FID_RESULTS.md](../../results/CelebA/expanded_fid_20260929/FID_RESULTS.md)
and [CSV](../../results/CelebA/expanded_fid_20260929/FID_RESULTS.csv)
distinguish complete results from planned counts for pending pools.
Training FIDs are 26.7717825405 (lower) and 23.9656050524 (upper), an
absolute difference of 2.8061774880. All additional samples and FID pools are
complete. Validation FIDs are 26.861024 (lower) and 25.318431 (upper), a gap
of 1.542593; final FIDs are 26.236221 and 25.303953, a gap of 0.932268.
See [the expanded main-result summary](../../docs/CelebA.md).

Generation array `56825939` contains four 20k sampling/feature tasks.
CPU report job `56825984` depended on successful completion of the entire
array. Both the generation array and report completed successfully.
The report records descriptive differences, with no formal equivalence test
or fixed acceptance cutoff.

The run's `plan.json`, `plan.sha256`, frozen `source/`, `inputs/`, and
`provenance/` record generator/source hashes, packages, Inception weights,
and the seed audit. Generated uint8 images and pool3 features are published
in atomic 1,000-image shard directories with hash receipts. Completed
20k pools receive `locks/extra_{role}_{branch}.json`. This supports verified
resumption without sampling different seeds after inspecting results.

FID uses raw canonical pytorch-fid 0.3.0 Inception pool3 features, pooled
float64 means and sample covariances (ddof=1). It does not use standardized
RDR features or average component-pool FIDs. All original input feature
hashes are checked against the historical locks.

To recompute completed pools using the exact saved source:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python3 -B /cwork/yx306/RDR/CelebA/expanded_fid_20260929/source/experiments/CelebA/expanded_fid_report.py \
  --output-dir /cwork/yx306/RDR/CelebA/expanded_fid_20260929 --threads 2
```

Six focused tests check pooled covariance against concatenated data,
sample covariance convention, source hash/metadata/count rejection, and
pending/complete reporting. Generation has also passed mocked completion,
resume, seed-overlap, and tamper checks. Scientific completion requires the
actual sample receipts and numeric reports, independently of these tests.
