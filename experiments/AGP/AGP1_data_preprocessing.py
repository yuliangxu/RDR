# %%
# AGP preprocessing: Credit to Ganchao Wei (https://weigcdsb.github.io/)
# ----- 0. Load library -----
import torch
import os
import sys
os.chdir("/hpc/home/yx306/RDR")


agp_data_path = "/hpc/group/mastatlab/yx306/AGP/data/"
ganchao_path = "/hpc/group/mastatlab/microbiome/clean/"
sys.path.append(ganchao_path)
import microbiome_unet as ganchao
import utils.AGP_help as agp_help



from importlib import reload
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from scipy.stats import describe
# from utils import *
import matplotlib.pyplot as plt

import os
import pickle
import numpy as np
import pandas as pd
from scipy.stats import describe
from keras.models import load_model
from sklearn.model_selection import train_test_split
import torch
# %% 
# 1 - download otu data, tree data
import zipfile
import urllib.request
from io import BytesIO, StringIO
url = "https://ftp.microbio.me/AmericanGut/latest/03-otus.zip"  # Use HTTPS if available, or adjust if needed
zip_content_otu = BytesIO()
with urllib.request.urlopen(url) as response:
    chunk_size = 1024 * 1024
    while True:
        chunk = response.read(chunk_size)
        if not chunk:
            break
        zip_content_otu.write(chunk)
zip_content_otu.seek(0)

with zipfile.ZipFile(zip_content_otu) as z:
    tree_path = "03-otus/100nt/gg-13_8-97-percent/97_otus.tree"
    with z.open(tree_path) as f:
        newick_str = f.read().decode('utf-8')
# %%
# 2. extract otu table
import h5py
import biom
with zipfile.ZipFile(zip_content_otu) as z:
    biom_path = "03-otus/100nt/gg-13_8-97-percent/otu_table.biom"
    
    # Open the BIOM file from the ZIP as binary, then load via h5py for HDF5 format
    with z.open(biom_path) as fb:
        with h5py.File(fb, 'r') as h5:
            # Load the BIOM table from the HDF5 file object
            otu_table = biom.load_table(h5)

# %%
# 3. extract taxa_list
otu_df = otu_table.to_dataframe(dense=True)
otu_ids = otu_table.ids(axis='observation')
taxonomy = otu_table.metadata(axis='observation')
taxa_list = ['; '.join(md['taxonomy']) if md and 'taxonomy' in md else str(otu_id) 
             for otu_id, md in zip(otu_ids, taxonomy)]
otu_df.index = taxa_list
otu_df.iloc[0:3, 0:5]

# %%
# 4. download covariate data
url = "http://ftp.microbio.me/AmericanGut/latest/04-meta.zip"  # Use HTTP as FTP may not work in browsers
zip_content_meta = BytesIO()
with urllib.request.urlopen(url) as response:
    while True:
        chunk = response.read(1024 * 1024)  # Read 1 MB at a time
        if not chunk:
            break
        zip_content_meta.write(chunk)
zip_content_meta.seek(0)  # Reset to the beginning for reading

with zipfile.ZipFile(zip_content_meta) as z:
    # Optional: List all files in the ZIP to inspect (uncomment if path needs verification)
    # print(z.namelist())
    
    # Updated path to the metadata file inside the ZIP (based on namelist)
    meta_path = "04-meta/ag-cleaned.txt"
    
    # Open the file as binary, then read as TSV with Pandas
    with z.open(meta_path) as f:
        meta_df = pd.read_table(f, sep='\t', dtype=str)  # Use str to avoid type issues; adjust as needed

meta_df.shape
# %%
# 5. filter data to match with meta data
otu_df = otu_df.groupby(level=0).sum()  # Unique taxonomy index now

# match sample id with meta data
col_idx1 = np.isin(otu_df.columns.values, meta_df.iloc[:,0].values)
otu_df2 = otu_df.iloc[:,col_idx1]

# correspondingly, filter the metadata
meta_df_filter = meta_df[meta_df.iloc[:,0].isin(otu_df2.columns.values)]

# %%
# 6. subsample the data 
samples_threshold = 5000
total_reads_per_sample = otu_df2.sum(axis=0)
abundance = otu_df2.loc[:,total_reads_per_sample>samples_threshold]
# Convert absolute counts into relative abundance (sum to 100).
abundance = abundance.apply(lambda x: x/x.sum()*100, axis=0)

# YX: keep only taxa with relative abundance > 10%
abundance = abundance.loc[ abundance.sum(axis=1) > 10 ,:]

# YX: keep only samples with zeros in fewer than 50% of taxa (all pass)
abundance = abundance.loc[ :, (abundance ==0).sum(axis=0) <= 0.9*abundance.shape[0]  ]
# Have mean abundance at least as high as the overall average.
abundance = abundance.loc[:, abundance.mean(axis =0) >= abundance.mean(axis =0).mean() ]

# match with meta data
metadata = meta_df_filter[meta_df_filter.iloc[:,0].isin(abundance.columns.values)]

# %%
# 7. load tree
import dendropy
with zipfile.ZipFile(zip_content_otu) as z:
    tree_path = "03-otus/100nt/gg-13_8-97-percent/97_otus.tree"
    with z.open(tree_path) as f:
        newick_str = f.read().decode('utf-8')
tree = dendropy.Tree.get(data=newick_str, schema='newick')

# subset the tree
taxa_to_otus = {}
for otu_id, tax_str in zip(otu_ids, taxa_list):
    if tax_str in abundance.index:  # Only kept unique taxonomy
        taxa_to_otus.setdefault(tax_str, []).append(otu_id)
repr_otu_ids = [min(otus, key=int) for otus in taxa_to_otus.values()]

tree.retain_taxa_with_labels(repr_otu_ids)
tree.resolve_polytomies(limit=2, update_bipartitions=False)

# %%
# 8. output variables used for LT transformation
taxa_dict_repr = {otu: next(tax for tax, otus in taxa_to_otus.items() if otu in otus) for otu in repr_otu_ids}
leaves = list(tree.leaf_node_iter())
for leaf in leaves:
    leaf.taxon.label = taxa_dict_repr[leaf.taxon.label]

leaves = sorted(leaves, key=lambda n: n.taxon.label)
for i, leaf in enumerate(leaves, 1):
    leaf.number = i

# Get internal nodes in postorder traversal
# internal_nodes = [node for node in tree.postorder_node_iter() if not node.is_leaf()]
internal_nodes = [node for node in tree.postorder_node_iter() if not node.is_leaf()][::-1]  # Reverse: root first

# Assign numbers to internal nodes starting from n_tips + 1
n_tips = len(leaves)
for i, node in enumerate(internal_nodes, n_tips + 1):
    node.number = i
# Build child_node_pd
data = []
index = []
for node in internal_nodes:
    children = node.child_nodes()
    if len(children) != 2:
        raise ValueError(f"Non-binary node found with {len(children)} children after resolution.")
    # Sort children by number for consistent order
    # Preserve original child order from tree file
    children_nodes = node.child_nodes()
    child1 = f"Node_{children_nodes[0].number}"
    child2 = f"Node_{children_nodes[1].number}"
    index.append(f"Node_{node.number}")
    data.append([child1, child2])

chid_node_pd = pd.DataFrame(data, columns=['Child.1', 'Child.2'], index=index)
tree_taxa_pd = pd.DataFrame([leaf.taxon.label for leaf in leaves], 
                            index=range(1, len(leaves) + 1), columns=[0])
out_names = abundance.index.values

# %%
# 9. Taxa name
def clean_tax_level(level_str):
    if '__' in level_str and level_str.endswith('__'):
        return '-'
    elif level_str.endswith('__;'):  # Rare edge case
        return '-'
    else:
        return level_str

# Split the taxonomy strings from relative_df.index into levels
taxa_split = []
for tax in abundance.index:
    levels = tax.split('; ')
    if len(levels) != 7:  # Safeguard if any anomalies
        levels = levels[:7] + ['-'] * (7 - len(levels))  # Pad if shorter
    cleaned_levels = [clean_tax_level(l) for l in levels]
    taxa_split.append(cleaned_levels)

# Create the DataFrame with split levels (7 taxonomy columns: V2 to V8)
taxa_df = pd.DataFrame(taxa_split, columns=['V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8'])

# Add the sequential ID column as V1 (starting from 1)
taxa_df.insert(0, 'V1', range(1, len(taxa_df) + 1))
taxa_df

# %%
# 10. Preprocess for training
from sklearn.model_selection import train_test_split
x1_all_raw = abundance.values.T
# xtrain_raw, xval_raw = train_test_split(x1_all_raw, test_size=0.2, random_state=1)
metadata_all = metadata.values
xtrain_raw, xval_raw, meta_train, meta_val = train_test_split(
    x1_all_raw, metadata_all, test_size=0.2, random_state=1
)

xtrain_raw = xtrain_raw/100
xval_raw = xval_raw/100
out_names = abundance.index.values
outTaxa_pd = pd.DataFrame(out_names)

# LT transform
child_node_matrix = chid_node_pd
taxa_names = tree_taxa_pd.values
out_names = outTaxa_pd.values

x_lt = ganchao.smooth_transformation(xtrain_raw,
                             child_node_matrix = child_node_matrix,
                             out_names = out_names.flatten(),
                             taxa_names = taxa_names.flatten(),
                             epsilon=1e-7,
                             noise_scale=0.) # takes 17s

x_recon = ganchao.reconstruct_lt_all(
    x_lt, 
    child_node_matrix,
    tree_taxa_pd,
    outTaxa_pd
) # takes 8m

# check difference
diff_abs = np.abs(x_recon - xtrain_raw)
print([np.min(diff_abs), np.max(diff_abs)])

# where is the max error?
i, k = np.unravel_index(np.argmax(np.abs(x_recon - xtrain_raw)), xtrain_raw.shape)
print("worst sample, leaf:", i, k, "orig=", xtrain_raw[i, k], "recon=", x_recon[i, k])




# # try ChatGPT's logistic tree transform
# reload(agp_help)
# # 1) Build topology (you already did this)
# child_l, child_r, leaf_order, internal_order, label_to_index = \
#     agp_help.build_topology_from_child_df(chid_node_pd, taxa_names=None)

# # 2) Map "Node_i" -> taxonomy string (from tree_taxa_pd where index = i)
# leaf_to_taxon = {f"Node_{i}": str(tree_taxa_pd.loc[i, 0]) for i in tree_taxa_pd.index}

# # 3) Map taxonomy string -> column index in your abundance matrix
# #    (use the exact table you'll transform; here I'm using xtrain_raw)
# out_names = abundance.T.columns.astype(str).to_numpy()
# name_to_out = {nm: i for i, nm in enumerate(out_names)}

# # 4) Build the indexer: for each leaf in tree order, which column is it?
# indexer = np.array([name_to_out[leaf_to_taxon[leaf]] for leaf in leaf_order], dtype=int)

# # one-hot
# K = xtrain_raw.shape[1]
# # sanity: indexer must be a permutation of 0..K-1
# assert sorted(indexer.tolist()) == list(range(K)), "indexer must be a permutation"

# # pick a random leaf j (tree order), then find its column position in X
# j = np.random.randint(K)
# pos = indexer[j]

# e = np.zeros((1, K), float)
# e[0, pos] = 1.0

# z_id  = agp_help.lt_forward_with_indexer(e, child_l, child_r, indexer)
# e_hat = agp_help.lt_back_anyorder_fixed(z_id, child_l, child_r, indexer, abs_to_j)

# print("one-hot max abs error:", np.max(np.abs(e - e_hat)))


# e = np.zeros((1, K)); e[0, np.random.randint(K)] = 1.0
# z_id = agp_help.lt_forward_with_indexer(e, child_l, child_r, indexer)
# e_hat = agp_help.lt_back_anyorder_fixed(z_id, child_l, child_r, indexer, abs_to_j)
# print("one-hot max abs error:", np.max(np.abs(e - e_hat)))

# # random Dirichlet
# rng = np.random.default_rng(0)
# xr = rng.dirichlet(np.ones(K), size=3)
# zr = lt_forward_with_indexer(xr, child_l, child_r, indexer)
# xr_hat = lt_back_anyorder_fixed(zr, child_l, child_r, indexer, abs_to_j)
# print("dirichlet max abs error:", np.max(np.abs(xr - xr_hat)))

# # your real data (make sure rows sum to 1)
# row_sums = xtrain_raw.sum(axis=1, keepdims=True)
# X = xtrain_raw / row_sums
# Z = agp_help.lt_forward_with_indexer(X, child_l, child_r, indexer)
# X_hat = agp_help.lt_back_anyorder_fixed(Z, child_l, child_r, indexer, abs_to_j)
# print("real-data max abs error:", np.max(np.abs(X - X_hat)))



# # 2) Map leaf label "Node_i" -> taxonomy string from tree_taxa_pd
# #    (tree_taxa_pd has index 1..n and a single column 0 with the taxonomy)
# leaf_to_taxon = {
#     f"Node_{i}": tree_taxa_pd.loc[i, 0]
#     for i in tree_taxa_pd.index
# }

# # 3) Map taxonomy -> column index in your abundance table
# out_names = abundance.index.values.astype(str)     # columns of your RA table
# name_to_out = {nm: i for i, nm in enumerate(out_names)}

# # 4) Build indexer to reorder RA columns into the tree *tip order*
# indexer = np.array([ name_to_out[ leaf_to_taxon[leaf] ] for leaf in leaf_order ], dtype=int)

# # 5) Forward / back (use the “with_indexer” versions I sent before)
# Z = agp_help.lt_forward_with_indexer(xtrain_raw, child_l, child_r, indexer) # 18.1s
# X_hat = agp_help.lt_back_with_indexer(Z, child_l, child_r, indexer) # 

# # 6) Check reconstruction error
# err = np.max(np.abs(xtrain_raw - X_hat))
# print("max abs recon error:", err)  # expect ~1e-12 ... 1e-9

# z = agp_help.lt_forward(xtrain_raw, child_l, child_r, out_names, taxa_names)
# x_hat = agp_help.lt_back(z, child_l, child_r, out_names, taxa_names)


# %%
# final. output data (n_taxa = 614, n_sample = 13054)

saveFolder =  "/hpc/group/mastatlab/yx306/AGP/data"

abundance.to_csv(saveFolder + '/yx1_abundance.csv', index=True)
np.savetxt(saveFolder + '/yx1_abundance_lt_train.csv', x_lt, delimiter=",")
np.savetxt(saveFolder + '/yx1_xtrain_raw.csv', xtrain_raw, delimiter=",")
np.savetxt(saveFolder + '/yx1_xtest_raw.csv', xval_raw, delimiter=",")
# x_lt = np.loadtxt(saveFolder + "x_lt.csv", delimiter=",")

# abundance.to_csv(saveFolder + '/yx1_xtrain_raw.csv', index=True)
# abundance.to_csv(saveFolder + '/yx1_abundance_train_lt.csv', index=True)
# abundance.to_csv(saveFolder + '/yx1_abundance_test.csv', index=True)

metadata.to_csv(saveFolder + '/yx2_metadata.csv', index=True)
meta_train_df = pd.DataFrame(meta_train, columns=metadata.columns)
meta_train_df.to_csv(saveFolder + '/yx2_metadata_train.csv', index=True)
meta_val_df = pd.DataFrame(meta_val, columns=metadata.columns)
meta_val_df.to_csv(saveFolder + '/yx2_metadata_test.csv', index=True)

chid_node_pd.to_csv(saveFolder + '/yx3_chid_node_df.csv', index=True)
tree_taxa_pd.to_csv(saveFolder + '/yx4_taxa_name_df.csv', index=True)
np.save(saveFolder + '/yx5_out_names.npy', out_names)
taxa_df.to_csv(saveFolder + '/yx6_taxa_df.csv', index=False)



