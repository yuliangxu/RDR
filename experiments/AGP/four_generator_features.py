"""Biological feature definitions used by the four-generator community tree.

These four functions are copied verbatim from the frozen downstream feature
implementation. They do not run the historical attribution/perturbation workflow.
"""
import numpy as np
import pandas as pd


def normalize(raw):
    x = np.asarray(raw, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all() or (x < 0).any():
        raise ValueError('Invalid composition')
    totals=x.sum(1, keepdims=True)
    if not np.isfinite(totals).all() or (totals <= 0).any():
        raise ValueError('Invalid composition totals')
    return x/totals


def build_groups(raw, taxa, k=12):
    x = normalize(raw)
    if len(taxa) != x.shape[1]:
        raise ValueError('Taxonomy dimension mismatch')
    if 'taxon_index' not in taxa or not np.array_equal(taxa.taxon_index.to_numpy(),np.arange(x.shape[1])):
        raise ValueError('Taxonomy coordinate order mismatch')
    groups = []
    columns = ['kingdom','phylum','class','order','family']
    for lineage, rows in taxa.groupby(columns, sort=True, dropna=False):
        if str(lineage[-1]) == 'Unclassified' or pd.isna(lineage[-1]):
            continue
        indices = rows.taxon_index.to_numpy(int)
        mass = x[:, indices].sum(1)
        groups.append(dict(name=str(lineage[-1]), lineage=list(map(str,lineage)), indices=indices.tolist(),
                           train_mean_abundance=float(mass.mean()), train_prevalence=float((mass>0).mean())))
    groups = sorted(groups, key=lambda g:(-g['train_mean_abundance'],g['lineage']))[:k]
    if len({g['name'] for g in groups}) != len(groups):
        for g in groups:
            g['name'] = ';'.join(g['lineage'])
    return groups


def feature_matrix(raw, groups):
    x = normalize(raw)
    positive = x>0
    logs = np.zeros_like(x)
    np.log(x, out=logs, where=positive)
    values = [positive.sum(1), -(x*logs).sum(1), x.max(1)]
    names = ['Richness', 'Shannon entropy', 'Maximum taxon abundance']
    masses = []
    for g in groups:
        m = x[:,g['indices']].sum(1)
        masses.append(m)
        values.extend([np.log((m+1e-6)/(1-m+1e-6)), (m>0).astype(float), positive[:,g['indices']].sum(1)])
        names.extend([g['name']+' log mass/rest',g['name']+' present',g['name']+' richness'])
    for i,g in enumerate(groups):
        for j in range(i):
            values.append(np.log((masses[i]+1e-6)/(masses[j]+1e-6)))
            names.append(g['name']+'/'+groups[j]['name']+' log balance')
    return np.column_stack(values), names


def rules(tree, names):
    result={}
    def visit(node, path):
        left,right=tree.tree_.children_left[node],tree.tree_.children_right[node]
        if left==right:
            result[str(node)]=' AND '.join(path) if path else 'All observations'
        else:
            name=names[tree.tree_.feature[node]]
            threshold=tree.tree_.threshold[node]
            visit(left,path+[f'{name} <= {threshold:.6g}'])
            visit(right,path+[f'{name} > {threshold:.6g}'])
    visit(0,[])
    return result
