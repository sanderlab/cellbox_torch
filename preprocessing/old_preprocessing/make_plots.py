import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr
from adjustText import adjust_text
import pandas as pd

df = pd.read_csv('prot_log.csv')
protein_expr = df.iloc[:, :-2]
targets = df["main_targets"]  # All columns except the last one
cell_viability = df["Cell_viability%_(cck8Drug-blk)/(control-blk)*100"]  # The last column
adata = ad.AnnData(
    X=protein_expr.values,  # The protein expression data as a numpy array
    var=pd.DataFrame(index=protein_expr.columns)  # The column names of protein expressions as variable (var) names
)
# Set the index of the obs DataFrame to the pert_id
adata.obs["Cell_Viability"] = list(cell_viability)
adata.obs["main_targets"] = list(targets)


adata.write('new_data.h5ad')

# data = adata.obs['Cell_Viability']
# plt.figure(figsize=(8, 6))
# plt.hist(data, bins=30, color='blue', edgecolor='black')
# plt.title(f'Histogram of Viabilities')
# plt.xlabel('Viabilities')
# plt.ylabel('Frequency')
# plt.grid(axis='y', alpha=0.75)
# plt.savefig('plots/z_viabilities_histogram.png')
# plt.close()

data = adata.obs['Cell_Viability']
plt.figure(figsize=(8, 6))
plt.hist(data, bins=30, color='blue', edgecolor='black')
plt.title(f'Histogram of Viabilities')
plt.xlabel('Viabilities')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.savefig('plots/viabilities_histogram.png')
plt.close()

viabilities = adata.obs['Cell_Viability']
correlation = np.corrcoef(adata.X)
sns.heatmap(correlation)
plt.savefig('plots/corr_matrix.png')
plt.close()

flat = correlation.flatten()
v_mat = np.repeat(viabilities, 23)
r, _ = pearsonr(v_mat, flat)
sns.scatterplot(y=flat, x=v_mat)
plt.title(f"Correlation and Viability, r = {round(r, 5)}")
plt.savefig('plots/Correlation_Viability.png')
plt.close()

flatty = adata.X.flatten()
v_matty = np.repeat(viabilities, 6139)
r, _ = pearsonr(v_matty, flatty)
sns.scatterplot(y=flatty, x=v_matty)
plt.title(f"Expression and Viability, r = {round(r, 5)}")
plt.savefig('plots/Expression_Viability.png')
plt.close()

sc.pp.neighbors(adata, metric='correlation', use_rep='X', n_neighbors=2)
sc.tl.umap(adata, min_dist=0.09)
plt.figure(figsize=(24, 18))
sc.pl.umap(adata, color=['Cell_Viability'], show=False, size=250)
plt.savefig('plots/nolabel_umap.png')
plt.close()

umap_coords = adata.obsm['X_umap']
perturbation_labels = adata.obs['main_targets']
fig, ax = plt.subplots(figsize=(15, 10))
sc.pl.umap(adata, color=['Cell_Viability'], show=False, size=500, ax=ax)
texts = []
for i, (x, y) in enumerate(umap_coords):
    target = adata.obs['main_targets'][i]
    texts.append(ax.text(x, y, str(target), fontsize=8))
adjust_text(texts, arrowprops=dict(arrowstyle='-', color='grey', lw=0.5), ax=ax)
plt.savefig('plots/label_umap.png')
plt.close()
