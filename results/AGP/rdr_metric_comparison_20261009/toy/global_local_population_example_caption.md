Exact population example: identical global summaries can conceal opposite locations of mismatch.
P assigns probability 0.5 to each of two communities, represented by scalar features -1 and +1.
Q1 assigns probabilities (0.8, 0.2), and Q2 assigns (0.2, 0.8).
A: the generators overrepresent opposite communities. B: their complete analytic precision–recall
for distributions (PRD) boundaries coincide because permuting the two states preserves P and
permutes Q. Both have population Gaussian Fréchet distance 0.4 in this scalar feature space.
An ideal deterministic two-label classifier would give both an Inception Score of exp(H(0.8, 0.2))
= 1.649385; no corresponding classifier or IS is claimed for AGP. C: population regional RDR is
10/13 and 10/7 for Q1 and reverses for Q2, identifying the underrepresented community.
These are exact population quantities, not finite-sample estimates; empirical metric estimates
will fluctuate. The Gaussian distance is not image/Inception-feature FID. The PRD statement is
about the aggregate PRD curve, not every precision–recall estimator or an individual-sample
realism/support diagnostic. Conditional metrics supplied with community labels can also locate
the imbalance. The example establishes the loss of location information in these particular
global summaries, not that RDR is the only possible local diagnostic.

PRD definition: Sajjadi et al. (2018), https://arxiv.org/abs/1806.00035.
Gaussian Fréchet/FID definition: Heusel et al. (2017), https://arxiv.org/abs/1706.08500.
IS definition: Salimans et al. (2016), https://arxiv.org/abs/1606.03498.
