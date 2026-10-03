"""Histogram score ranges must respect equal source weights and raw scores."""
import numpy as np
import pytest
from experiments.CelebA.publication_figures import balanced_quantiles


def test_unequal_fold_sizes_have_equal_mixture_weight():
    a=np.zeros(1);b=np.full(99,2.)
    np.testing.assert_array_equal(balanced_quantiles(a,b,[.025,.49,.51,.975]),[0,0,2,2])
    # The unweighted pool would incorrectly discard the small fold here.
    assert np.quantile(np.r_[a,b],.025)==2


def test_repeated_values_endpoints_and_fold_exchange():
    a=np.array([0.,.9,1.,1.,2.]);b=np.array([.8,1.,1.2])
    np.testing.assert_array_equal(balanced_quantiles(a,b),balanced_quantiles(b,a))
    np.testing.assert_array_equal(balanced_quantiles(a,b,[0,1]),[0,2])
    np.testing.assert_array_equal(balanced_quantiles(np.ones(4),np.ones(9)),[1,1])


@pytest.mark.parametrize('a,b',[(np.array([]),np.ones(2)),(np.array([np.nan]),np.ones(2)),(np.array([2.1]),np.ones(2))])
def test_invalid_scores_do_not_get_clipped_or_replaced(a,b):
    with pytest.raises(ValueError):balanced_quantiles(a,b)
