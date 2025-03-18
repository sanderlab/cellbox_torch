import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import anndata as ad
from scipy.stats import zscore
sns.set_theme()

df = pd.read_csv("data.csv", low_memory=False)
nan_thres = 0.05
var_thres = 0.1
targets_vector = df['target'].drop(index=84).values


# Make New dataframe with pert_id as row index and Cell_viability as the only concatenated column
cols = df.columns.tolist()[1:-16] + ["pert_id", "Cell_viability%_(cck8Drug-blk)/(control-blk)*100"]
df = df[cols].set_index("pert_id")
df.index.name = 'pert_id'


# Identify columns with > nan_thres percentage of Nan values
nan_percentage = df.isna().mean()
prots_less_than_some_nan = nan_percentage[nan_percentage < nan_thres].index.tolist()

# Identify columns that have NaN in control
control_row = df.loc['control']
prots_control_not_nan = control_row.notna().index.tolist()

# Identify proteins with lower variance threshold across conditions including control
variances = df[df.columns[:-1]].std(axis=0).to_frame().rename(columns={0: "std"})
means = df[df.columns[:-1]].mean(axis=0).to_frame().rename(columns={0: "mean"})
mean_var_df = variances.merge(means, right_index=True, left_index=True)
mean_var_df['dispersion'] = mean_var_df["std"] / mean_var_df["mean"]
prots_highly_var = mean_var_df[mean_var_df["dispersion"] > 0.01].index.tolist()

# Retain the proteins
prots_retained = list(
    set(prots_less_than_some_nan)
    .intersection(set(prots_control_not_nan))
    .intersection(set(prots_highly_var))
)

# Make new dataframe with filtered proteins and cell viability
df = df[prots_retained + ["Cell_viability%_(cck8Drug-blk)/(control-blk)*100"]]

# SimpleImputer with mean without control sample
df_tar_no_control = df[df.index != "control"]
control = df.loc[["control"], :]

from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
np_tar = imp.fit_transform(df)
df = pd.DataFrame(np_tar, columns=df.columns, index=df.index)

# Divide the control to all other samples, then take the log
control = df.loc[["control"]]
prot = df[df.index != 'control']

# Transform the data by log2(pert/control)
prot_log = (prot).div(control.squeeze(), axis=1)
prot_log = np.log2(prot_log)


def normalize_and_scale(vector):
    """Normalize and standardize a vector."""
    # Normalize the vector to the range [0, 1]
    min_val = np.min(vector)
    max_val = np.max(vector)
    normalized_vector = (vector - min_val) / (max_val - min_val)

    # Standardize the normalized vector to have mean 0 and standard deviation 1
    mean = np.mean(normalized_vector)
    std_dev = np.std(normalized_vector)
    standardized_vector = (normalized_vector - mean) / std_dev

    return standardized_vector

viabilities = prot_log["Cell_viability%_(cck8Drug-blk)/(control-blk)*100"]
ns_viabilities = normalize_and_scale(list(viabilities))

var_names = prot_log.columns[:-1]

adata = ad.AnnData(X=prot_log.iloc[:, :-1].values, var=pd.DataFrame(index=var_names))
adata.obs['pert_id'] = prot_log.index
adata.obs['z_viabilities'] = ns_viabilities
adata.obs['viabilities'] = list(viabilities)
adata.obs['targets'] = targets_vector

adata.write('processed_data.h5ad')
