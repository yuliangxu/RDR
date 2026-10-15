# Four-generator PCoA colored by calibration CP intervals

![Discrete CP interval categories](pcoa_cp_categories.png)

This adds categorical colors to the existing joint Bray–Curtis PCoA. All
10,358 observations, coordinates, axis limits, and the Lingoes correction are
unchanged. Real P is the top row; generated Q is the bottom row.

Each point uses its pair-specific **final fit 0** score to enter one of the
original 20 fixed-width bins on [0,2]. The saved bin's **C.2 Clopper–Pearson
interval** determines the color: orange if its upper bound is below 1, blue
if its lower bound is above 1, and gray if it contains 1 (including endpoints).
Orange denotes generated-mass excess in that score cell; blue denotes a
deficit. Gray means the interval does not resolve a direction, not equality.

These are the same nominal 95% intervals used in the fit-0 calibration panel:
the CP tail allocation is 0.05/(4*20), simultaneous over 20 cells within one
fit, not across methods or initializations. They describe cell-average RDR
2P(A)/(P(A)+Q(A)), not confidence intervals for individual true RDR values.
The assessment observations also supplied the cell counts; interpretation
remains retrospective and nominal under the original sampling limitations.
No intervals or models were refitted and no interval bounds were averaged.
The preceding continuous-color plot instead uses the five-fit mean score.

The displayed plane contains only 0.3121% of Lingoes-corrected inertia.
Projected overlap does not establish equality of the full distributions.

[PDF](pcoa_cp_categories.pdf) · [Category counts](category_counts.csv) ·
[Saved calibration cells](calibration_cells.csv) ·
[Rendering source](source/four_generator_pcoa_cp.py) · [Manifest](manifest.json)

The manifest pins input hashes and SHA-256 hashes of the eight ordered uint8
category vectors (0 below, 1 contains, 2 above). Row-level coordinates and
identities remain in the original private ordination archive.
