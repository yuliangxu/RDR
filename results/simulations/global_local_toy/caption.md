Identical global summaries, opposite local mismatch. (a) Discrete probability
distributions P, Q1, and Q2 have common support {-1,+1} and probability mass
functions p, q1, and q2. In the order x=-1,+1, their probability masses are
(0.5,0.5), (0.8,0.2), and (0.2,0.8), respectively.
Equivalently, X=2B-1 for B~Bernoulli(theta), with theta=0.5,0.2,0.8.
(b) The entire population precision-recall-for-distributions (PRD) curves
coincide. Both generators have scalar-feature Gaussian Frechet distance
(the FID formula applied to these features) 0.4, and Inception Score 1.649385
under an ideal deterministic classifier that assigns distinct labels to -1 and +1.
(c) Exact RDR, r_k(x)=2p(x)/(p(x)+q_k(x)), is (10/13,10/7) for Q1 and
(10/7,10/13) for Q2, in the same support order. RDR below one means q_k(x)>p(x);
RDR above one means q_k(x)<p(x). The dotted line marks equal probability mass.

The global metrics detect mismatch here but do not identify its location.
These are exact population quantities, not empirical image/Inception metrics
or estimated RDR. The PRD claim concerns the population PRD definition of
Sajjadi et al.; other conditional or local metrics can also locate imbalance.
Finite-sample estimates need not coincide. The unchanged construction was
originally reported in results/AGP/rdr_metric_comparison_20261009/toy/.

PRD: https://arxiv.org/abs/1806.00035
Gaussian Frechet/FID: https://arxiv.org/abs/1706.08500
IS: https://arxiv.org/abs/1606.03498
