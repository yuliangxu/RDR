"""Biological feature and rule contracts used by the shared community map."""
import numpy as np
from sklearn.tree import DecisionTreeRegressor

from experiments.AGP import four_generator_features as features
from experiments.AGP import rdr_downstream as frozen_compatibility


def test_family_features_preserve_compositional_scale_and_zero_presence():
    raw = np.array([[2., 0., 3., 0.], [0., 4., 0., 0.]])
    groups = [dict(name="Family A", indices=[0, 1]),
              dict(name="Family B", indices=[2, 3])]
    values, names = features.feature_matrix(raw, groups)
    rescaled, same_names = features.feature_matrix(raw * np.array([[7.], [3.]]), groups)
    np.testing.assert_allclose(values, rescaled)
    assert names == same_names
    columns = dict(zip(names, values.T))
    np.testing.assert_array_equal(columns["Richness"], [2, 1])
    np.testing.assert_allclose(columns["Shannon entropy"],
                               [-.4*np.log(.4)-.6*np.log(.6), 0.])
    np.testing.assert_allclose(columns["Maximum taxon abundance"], [.6, 1.])
    np.testing.assert_array_equal(columns["Family B present"], [1, 0])
    np.testing.assert_array_equal(columns["Family B richness"], [1, 0])
    np.testing.assert_allclose(columns["Family B/Family A log balance"],
                               np.log((np.array([.6, 0.])+1e-6)/(np.array([.4, 1.])+1e-6)))


def test_rule_directions_match_tree_leaves_and_frozen_imports():
    tree = DecisionTreeRegressor(max_depth=1, random_state=0)
    tree.fit(np.arange(4.).reshape(-1, 1), [0., 0., 1., 1.])
    rules = features.rules(tree, ["Family mass"])
    left, right = tree.apply([[0.], [3.]])
    assert rules[str(left)] == "Family mass <= 1.5"
    assert rules[str(right)] == "Family mass > 1.5"
    assert frozen_compatibility.rules(tree, ["Family mass"]) == rules
