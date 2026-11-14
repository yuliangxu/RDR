"""Analytic targets, source weighting, and stable logit/BCE checks for JS(P,Q)."""
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments/AGP'))
from js_pq import evaluate_js_pq


class VariationalJSTests(unittest.TestCase):
    def test_identical_distributions(self):
        for representation, value in (('rdr', 1), ('logits', 0)):
            result = evaluate_js_pq([value]*7, [value]*13, representation=representation)
            self.assertAlmostEqual(result['js_p_q_variational'], 0, places=15)
            self.assertEqual((result['n_p'], result['n_q']), (7, 13))

    def test_known_discrete_target_with_unequal_cohort_sizes(self):
        # P=(4/5,1/5), Q=(1/5,4/5), midpoint=(1/2,1/2).
        p, q = [1.6]*8 + [.4]*2, [1.6]*3 + [.4]*12
        target = .8*math.log(1.6) + .2*math.log(.4)
        self.assertAlmostEqual(evaluate_js_pq(p, q)['js_p_q_variational'], target, places=14)
        tp, tq = ([math.log(r/(2-r)) for r in source] for source in (p, q))
        self.assertAlmostEqual(evaluate_js_pq(tp, tq, representation='logits')['js_p_q_variational'], target, places=14)

    def test_asymmetric_discrete_distribution(self):
        # P=(9/10,1/10), Q=(1/2,1/2); different P/Q log means.
        p_mass, q_mass = [.9, .1], [.5, .5]
        r = [2*p/(p+q) for p, q in zip(p_mass, q_mass)]
        target = .5*sum(p*math.log(2*p/(p+q)) + q*math.log(2*q/(p+q))
                        for p, q in zip(p_mass, q_mass))
        p, q = [r[0]]*9 + [r[1]], [r[0]]*7 + [r[1]]*7
        self.assertAlmostEqual(evaluate_js_pq(p, q)['js_p_q_variational'], target, places=14)
        # Replicating just one cohort must not alter equal source weights.
        self.assertAlmostEqual(evaluate_js_pq(p*3, q)['js_p_q_variational'], target, places=14)

    def test_disjoint_support_observed_endpoints(self):
        self.assertEqual(evaluate_js_pq([2]*3, [0]*7)['js_p_q_variational'], math.log(2))

    def test_negative_estimate_is_retained(self):
        result = evaluate_js_pq([.1]*5, [1.9]*9)
        self.assertLess(result['js_p_q_variational'], 0)
        self.assertAlmostEqual(result['js_p_q_variational'], math.log(.1), places=14)

    def test_independently_computed_balanced_bce(self):
        p, q = [-3., -.4, 2.], [-2., .3, 1., 4., 7.]
        # Independent moderate-logit probability calculation.
        prob_p, prob_q = ([1/(1+math.exp(-t)) for t in source] for source in (p, q))
        bce = -.5*sum(math.log(x) for x in prob_p)/len(p) \
              -.5*sum(math.log1p(-x) for x in prob_q)/len(q)
        result = evaluate_js_pq(p, q, representation='logits')
        self.assertAlmostEqual(result['js_p_q_variational'], math.log(2)-bce, places=13)
        self.assertAlmostEqual(result['balanced_bce'], bce, places=13)

    def test_extreme_logits_without_overflow_or_clipping(self):
        positive = evaluate_js_pq([1000.], [-1000.]*2, representation='logits')
        negative = evaluate_js_pq([-1000.], [1000.]*2, representation='logits')
        self.assertEqual(positive['js_p_q_variational'], math.log(2))
        self.assertAlmostEqual(negative['js_p_q_variational'], math.log(2)-1000, places=12)

    def test_endpoint_can_be_recovered_with_logit(self):
        # expit(40) rounds to one in float64, but the Q log remains finite.
        result = evaluate_js_pq([0.], [40.], representation='logits')
        self.assertTrue(math.isfinite(result['js_p_q_variational']))
        self.assertAlmostEqual(result['mean_log_two_minus_r_q'], math.log(2)-40, places=13)

    def test_invalid_or_unrecoverable_inputs_fail(self):
        cases = [([], [1]), ([1], []), ([math.nan], [1]), ([1], [math.inf]),
                 ([-.1], [1]), ([1], [2.1]), ([0], [1]), ([1], [2])]
        for p, q in cases:
            with self.subTest(p=p, q=q), self.assertRaises(ValueError):
                evaluate_js_pq(p, q)
        with self.assertRaisesRegex(ValueError, 'representation'):
            evaluate_js_pq([1], [1], representation='unknown')


if __name__ == '__main__':
    unittest.main()
