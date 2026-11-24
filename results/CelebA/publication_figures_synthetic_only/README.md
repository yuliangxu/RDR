# JRSSB Figures 7 and S5 · Synthetic CelebA-64 portraits

| Panel | Representation | Truncation | Figure | Preview |
| --- | --- | --- | --- | --- |
| 7A | Inception pool3 features | Lower, $Q_l$ | [PDF](images_feature_lower_ranked.pdf) | [PNG](images_feature_lower_ranked.png) |
| 7B | Inception pool3 features | Upper, $Q_u$ | [PDF](images_feature_upper_ranked.pdf) | [PNG](images_feature_upper_ranked.png) |
| S5A | RGB pixels | Lower, $Q_l$ | [PDF](images_pixel_lower_ranked.pdf) | [PNG](images_pixel_lower_ranked.png) |
| S5B | RGB pixels | Upper, $Q_u$ | [PDF](images_pixel_upper_ranked.pdf) | [PNG](images_pixel_upper_ranked.png) |

## Replacement captions

**Figure 7.** Synthetic CelebA-64 portraits from Diffusion-StyleGAN2 at lower (A, $Q_l$) and upper (B, $Q_u$) truncation. From left to right, each montage shows the 40 generated portraits whose repeat-0 RDRs fitted on Inception pool3 features are smallest, closest to 1, or largest. Brackets give the observed score range.

**Figure S5.** Synthetic CelebA-64 portraits at lower (A, $Q_l$) and upper (B, $Q_u$) truncation, ranked by repeat-0 RDR fitted on RGB pixels. Each montage shows 40 generated portraits in the saved display order. In A, the scores of the 40 portraits closest to 1 span [0, 2] at the displayed precision.

## Reproduce

Run from the repository root, with NumPy, Matplotlib and PyMuPDF 1.26.5 installed. Choose a new output directory.

```bash
python3 -m experiments.CelebA.synthetic_ranked_figures \
  --output /tmp/celeba-synthetic-ranked
```

The public command reads the generated-row image objects from the [sealed reference figures](../publication_figures_20261003_v2/FIGURES.md). It checks the two rows and three column positions, recovers the native RGB pixels, and builds new one-page PDFs. For this release the command also used the following option to read the original generated arrays and verify each displayed portrait's ID, row, role, and fitted score.

```text
--archive /cwork/yx306/RDR/CelebA/final_evaluation_expanded_20261003
```

The [selection manifest](selected_images.csv) preserves all 480 display entries in their original rank order across 12 groups (468 distinct source IDs). The [range table](image_ranges.csv) retains full precision. Each PDF contains three embedded RGB images in left-to-right group order. Each is a 4 × 10 montage of 64 × 64 generated portraits. The [verification record](verification.json) records the source object positions, native pixel hashes, output objects and file hashes. PNGs are 300 dpi renders of the delivered PDFs at a page width of seven inches.

## Existing public links showing real portraits

The following galleries on `updated` still link to the original two-row ranked panels, which include real portraits in the top row.

- [CelebA report, image gallery and links](https://github.com/yuliangxu/RDR/blob/updated/docs/CelebA.md#L162-L175)
- [Merged-test report, ranked gallery](https://github.com/yuliangxu/RDR/blob/updated/results/CelebA/merged_test_20261003/RESULTS.md#L36-L46) and [figure-file list](https://github.com/yuliangxu/RDR/blob/updated/results/CelebA/merged_test_20261003/RESULTS.md#L60-L67)
- [Original publication gallery, four PNG and PDF pairs](https://github.com/yuliangxu/RDR/blob/updated/results/CelebA/publication_figures_20261003_v2/FIGURES.md#L27-L37)

The separate all-bin diagnostic PDFs listed alongside the ranked panels in the CelebA and merged-test reports also include real portraits.
