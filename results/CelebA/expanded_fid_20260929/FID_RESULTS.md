# CelebA expanded-pool FID

**Status: complete.**

Raw pytorch-fid 0.3.0 pool3 features; float64 pooled moments and sample covariance (ddof=1). The same P pool is used for both generator comparisons within each role. Final combines the historical design and test pools. Validation/final Q add 20,000 new images per branch.

| Role | Status | P count | Q count per branch | FID lower | FID upper | Upper minus lower | Absolute difference |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | complete | 122984 | 120000 | 26.771783 | 23.965605 | -2.806177 | 2.806177 |
| validation | complete | 39786 | 40000 | 26.861024 | 25.318431 | -1.542593 | 1.542593 |
| final | complete | 39829 | 40000 | 26.236221 | 25.303953 | -0.932268 | 0.932268 |

Counts in pending rows are planned targets. These empirical FID differences are descriptive; they do not establish formal FID equivalence. Historical data and generator choices were inspected previously, so this is a retrospective comparison. FID is separate from RDR Brier/local-Gap model selection.

[Numeric results](FID_RESULTS.csv) · [Train provenance](fid/train.json) · [Validation provenance](fid/validation.json) · [Final provenance](fid/final.json)
