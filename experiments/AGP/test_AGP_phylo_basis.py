"""Alignment and compositional-coordinate checks for the AGP tree loader."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

try:
    from .AGP_phylo_basis import load_phylo_basis
except ImportError:
    from AGP_phylo_basis import load_phylo_basis


class PhyloBasisTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        # Deliberately permuted abundance columns and internal-node row order.
        self.names = ["C", "A", "D", "B"]
        pd.DataFrame({"sample": [.2, .1, .4, .3]}, index=self.names).to_csv(self.root / "yx1_abundance.csv")
        np.save(self.root / "yx5_out_names.npy", np.array(self.names, dtype=object)[:, None])
        pd.DataFrame({"0": ["A", "B", "C", "D"]}, index=[1, 2, 3, 4]).to_csv(self.root / "yx4_taxa_name_df.csv")
        self.children = pd.DataFrame(
            [["Node_3", "Node_4"], ["Node_6", "Node_7"], ["Node_1", "Node_2"]],
            columns=["Child.1", "Child.2"], index=["Node_7", "Node_5", "Node_6"],
        )
        self.children.to_csv(self.root / "yx3_chid_node_df.csv")

    def test_geometric_balances_alignment_and_inverse(self):
        basis, metadata = load_phylo_basis(self.root)
        x = np.array([.2, .1, .4, .3])  # C, A, D, B
        z = np.log(x) @ basis.T
        expected = [np.log(.2 / .4) / np.sqrt(2),
                    np.log(np.sqrt(.1 * .3) / np.sqrt(.2 * .4)),
                    np.log(.1 / .3) / np.sqrt(2)]
        assert_allclose(z, expected, atol=1e-14)
        recovered = np.exp(z @ basis)
        assert_allclose(recovered / recovered.sum(), x, atol=1e-14)
        assert_allclose(np.log(13.7 * x) @ basis.T, z, atol=1e-14)
        self.assertEqual(metadata["root_node"], "Node_5")
        self.assertEqual(basis.shape, (3, 4))

    def test_generator_order_mismatch_rejected(self):
        np.save(self.root / "yx5_out_names.npy", np.array(sorted(self.names)))
        with self.assertRaisesRegex(ValueError, "Generator output taxon order"):
            load_phylo_basis(self.root)

    def test_repeated_leaf_rejected(self):
        self.children.loc["Node_6", "Child.2"] = "Node_1"
        self.children.to_csv(self.root / "yx3_chid_node_df.csv")
        with self.assertRaisesRegex(ValueError, "one root"):
            load_phylo_basis(self.root)

    def test_disconnected_cycle_rejected(self):
        self.children.loc["Node_5"] = ["Node_1", "Node_2"]
        self.children.loc["Node_6"] = ["Node_7", "Node_3"]
        self.children.loc["Node_7"] = ["Node_6", "Node_4"]
        self.children.to_csv(self.root / "yx3_chid_node_df.csv")
        with self.assertRaisesRegex(ValueError, "disconnected"):
            load_phylo_basis(self.root)


if __name__ == "__main__":
    unittest.main()
