"""Unweighted phylogenetic ILR basis from the archived AGP tree.

For positive compositions X in the archived abundance-column order, compute
``np.log(X) @ basis.T``.  A row is the scaled difference of the geometric-mean
log abundances of the two child clades, not the generator's clade-sum logit.
Zero handling belongs to the caller and must be identical for P and Q.
"""

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_phylo_basis(data_root):
    """Return a float64 (D-1, D) basis and JSON-compatible provenance.

    The archived ``yx5_out_names.npy`` is a trusted, locally generated object
    array of strings; loading it permits pickle to check the generator's output
    taxon order against the real abundance table.  The tree must be a connected,
    rooted binary tree with every named taxon appearing exactly once.
    """
    root = Path(data_root).resolve()
    paths = {
        "abundance": root / "yx1_abundance.csv",
        "children": root / "yx3_chid_node_df.csv",
        "leaves": root / "yx4_taxa_name_df.csv",
        "generator_taxon_order": root / "yx5_out_names.npy",
    }
    names = pd.read_csv(
        paths["abundance"], usecols=[0], dtype=str, keep_default_na=False,
    ).iloc[:, 0].tolist()
    dimension = len(names)
    if dimension < 2 or len(set(names)) != dimension or any(not x for x in names):
        raise ValueError("Abundance taxa must be unique, nonempty, and number at least two.")
    saved_names = np.load(paths["generator_taxon_order"], allow_pickle=True)
    if saved_names.shape not in [(dimension,), (dimension, 1)]:
        raise ValueError("Unexpected archived generator taxon-order shape.")
    if saved_names.reshape(-1).tolist() != names:
        raise ValueError("Generator output taxon order differs from the abundance table.")

    children = pd.read_csv(paths["children"], index_col=0, dtype=str, keep_default_na=False)
    leaves = pd.read_csv(paths["leaves"], index_col=0, dtype=str, keep_default_na=False)
    if list(children.columns) != ["Child.1", "Child.2"] or not children.index.is_unique:
        raise ValueError("The child table must have unique parent IDs and two named child columns.")
    if leaves.shape != (dimension, 1) or not leaves.index.is_unique:
        raise ValueError("The leaf table must contain one unique node ID per abundance taxon.")
    leaf_ids = [f"Node_{index}" for index in leaves.index]
    leaf_names = leaves.iloc[:, 0].tolist()
    if len(set(leaf_names)) != dimension or set(leaf_names) != set(names):
        raise ValueError("Tree leaves do not match abundance taxa one-to-one.")
    name_to_column = {name: column for column, name in enumerate(names)}
    leaf_to_column = {node: name_to_column[name] for node, name in zip(leaf_ids, leaf_names)}
    child_map = {str(node): tuple(row) for node, row in zip(children.index, children.to_numpy())}
    internal_ids = list(child_map)
    if len(internal_ids) != dimension - 1 or set(child_map) & set(leaf_ids):
        raise ValueError("A binary tree needs D-1 distinct internal nodes, disjoint from its leaves.")
    all_nodes = set(child_map) | set(leaf_ids)
    parent_counts = Counter(child for pair in child_map.values() for child in pair)
    if set(parent_counts) - all_nodes:
        raise ValueError("The tree references unknown child nodes.")
    roots = all_nodes - set(parent_counts)
    if len(roots) != 1 or any(count != 1 for count in parent_counts.values()):
        raise ValueError("The tree must have one root and exactly one parent for every other node.")
    root_node = next(iter(roots))
    if root_node not in child_map:
        raise ValueError("The root must be an internal node.")

    # Iterative postorder avoids a recursion-depth assumption about the tree.
    descendants = {node: np.array([column], dtype=np.int64)
                   for node, column in leaf_to_column.items()}
    entered = set()
    stack = [(root_node, False)]
    while stack:
        node, closing = stack.pop()
        if closing:
            left, right = child_map[node]
            descendants[node] = np.concatenate((descendants[left], descendants[right]))
            continue
        if node in entered:
            raise ValueError("The tree contains a repeated node or cycle.")
        entered.add(node)
        if node in child_map:
            left, right = child_map[node]
            stack.extend([(node, True), (right, False), (left, False)])
    if entered != all_nodes or sorted(descendants[root_node].tolist()) != list(range(dimension)):
        raise ValueError("The tree is disconnected or does not cover every taxon exactly once.")

    basis = np.zeros((dimension - 1, dimension), dtype=np.float64)
    balances = []
    for row, node in enumerate(internal_ids):
        left_node, right_node = child_map[node]
        left, right = descendants[left_node], descendants[right_node]
        r, s = len(left), len(right)
        basis[row, left] = np.sqrt(s / (r * (r + s)))
        basis[row, right] = -np.sqrt(r / (s * (r + s)))
        balances.append({"node": node, "left_child": left_node, "right_child": right_node,
                         "left_taxon_count": r, "right_taxon_count": s})
    orthogonality_error = float(np.max(np.abs(basis @ basis.T - np.eye(dimension - 1))))
    zero_sum_error = float(np.max(np.abs(basis.sum(axis=1))))
    if orthogonality_error > 1e-12 or zero_sum_error > 1e-12:
        raise ValueError("The constructed balance basis failed its orthonormal/zero-sum checks.")
    metadata = {
        "method": "unweighted_phylogenetic_ilr",
        "definition": "sqrt(r*s/(r+s)) * log(geometric_mean(left)/geometric_mean(right))",
        "input_dimension": dimension,
        "output_dimension": dimension - 1,
        "root_node": root_node,
        "taxon_names_in_input_order": names,
        "leaf_nodes_in_input_order": [node for node, _ in sorted(leaf_to_column.items(), key=lambda x: x[1])],
        "balances_in_output_order": balances,
        "max_orthogonality_error": orthogonality_error,
        "max_zero_sum_error": zero_sum_error,
        "generator_taxon_order_matches_abundance": True,
        "basis_sha256": hashlib.sha256(basis.tobytes(order="C")).hexdigest(),
        "taxon_order_sha256": hashlib.sha256(json.dumps(names, ensure_ascii=False).encode()).hexdigest(),
        "source_files": {key: {"path": str(path), "sha256": _sha256(path)} for key, path in paths.items()},
        "weights": "No taxon or branch-length weighting.",
        "zero_handling": "None in this loader; the caller supplies positive compositions.",
        "tree_provenance": (
            "Archived AGP preprocessing pruned the Greengenes 13_8 97-percent OTU tree to one "
            "representative OTU (minimum numeric ID) per retained taxonomy group, then resolved "
            "polytomies to binary nodes. This loader preserves those archived child links; it "
            "does not reconstruct branch lengths or unresolved branching uncertainty."
        ),
    }
    return basis, metadata
