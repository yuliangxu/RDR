# Four-generator Bray–Curtis PCoA

![Assessment samples colored by source](pcoa_sources.png)

One joint map includes all 10,358 assessment observations: 2,002 real samples
once, plus 2,089 generated samples from each retained generator. All panels
use the same coordinates and axes. No samples are removed or subsampled.

![The same map colored by fitted RDR](pcoa_rdr.png)

Rows show real P and generated Q separately. Colors use each P-versus-Q pair's
mean fitted RDR over its five completed final initializations, on the same
0–2 scale. The real coordinates are repeated unchanged across columns; their
colors differ because each generator has a separately selected evaluator.
Orange scores below one indicate fitted generated-mass excess; blue scores
above one indicate fitted generated-mass deficit. These are fitted scores,
with the calibration limitations documented in the main AGP report.

Bray–Curtis is computed after row closure in float64. Lingoes correction uses
c = 22.02027998. PCo 1 and PCo 2 explain
0.1705% and 0.1416% of corrected inertia, respectively
(0.3121% together). Projected overlap does not establish equality
of the full distributions, and this descriptive assessment map was not used
for generator fitting, checkpoint choice, RDR selection, or community discovery.

Exact pairwise distances are used. Deterministic iterative eigensolves estimate
the two largest and the smallest eigenpairs; residuals and corrected total
inertia are retained in [diagnostics](ordination_diagnostics.json). No full
eigenvalue spectrum or raw negative-inertia fraction is claimed.

[Source-colored PDF](pcoa_sources.pdf) · [RDR-colored PDF](pcoa_rdr.pdf) ·
[Rendering source](source/four_generator_pcoa.py)

The four sealed studies and their scores are unchanged. Input hashes, private
coordinates, and row identities stay under `/cwork/yx306/RDR/AGP/four_generator_pcoa_20261014`; public outputs contain
figures, aggregate diagnostics, and the rendering source.
