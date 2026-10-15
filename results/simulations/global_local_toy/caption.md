Identical global summaries, opposite local mismatch. (a) On states A=-1
and B=+1, P=(0.5,0.5), Q1=(0.8,0.2), and Q2=(0.2,0.8).
(b) The entire population precision-recall-for-distributions (PRD) curves
coincide. Both generators have scalar-feature Gaussian Frechet distance
(the FID formula applied to these features) 0.4, and Inception Score 1.649385
under an ideal deterministic classifier that labels the two states.
(c) Exact RDR, r_k(x)=2p(x)/(p(x)+q_k(x)), is (10/13,10/7) for Q1 and
(10/7,10/13) for Q2. RDR below one identifies overrepresentation by Q;
RDR above one identifies underrepresentation. The dotted line marks balance.

The global metrics detect mismatch here but do not identify its location.
These are exact population quantities, not empirical image/Inception metrics
or estimated RDR. The PRD claim concerns the population PRD definition of
Sajjadi et al.; other conditional or local metrics can also locate imbalance.
Finite-sample estimates need not coincide. The unchanged construction was
originally reported in results/AGP/rdr_metric_comparison_20261009/toy/.

PRD: https://arxiv.org/abs/1806.00035
Gaussian Frechet/FID: https://arxiv.org/abs/1706.08500
IS: https://arxiv.org/abs/1606.03498
