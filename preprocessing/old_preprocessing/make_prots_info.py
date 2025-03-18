# NEED PANDAS DATAFRAME WITH index= proteins, columns = nan_prop_samples_with_targets,
# NaN_prop_all_samples, target, NaN_control, targeting_pert_IDs

import pandas as pd
import numpy as np

# Sample data (replace with your actual data)
df = pd.read_csv('data.csv')
df['Uniprot.ID'] = df['Uniprot.ID'].astype(str)
df = df.drop('sample_id', axis=1)

# List of invalid Uniprot IDs
invalid_uniprot_ids = ["not protein", "not human", 'nan']

# Identify perturbations with valid targets
df_valid_targets = df[~df['Uniprot.ID'].isin(invalid_uniprot_ids)]

# Calculate the proportion of NaN values for proteins in perturbations with targets
nan_prop_samples_with_targets = df_valid_targets.iloc[:, :-12].isna().mean()

# Calculate the proportion of NaN values for all samples
nan_prop_all_samples = df.iloc[:, :-12].isna().mean()

# Determine if there are NaN values for each protein in the control sample
control_sample = df[df['pert_id'] == 'control']
nan_control = control_sample.iloc[:, :-12].isna().iloc[0]

# Identify proteins that are targets of any perturbation
target_proteins = df_valid_targets['Uniprot.ID'].unique()

def split_by_semi(string):
    out = []
    id = ''
    for char in string:
        if char != ';':
            id += char
        else:
            out.append(id)
            id = ''
    out.append(id)
    return out

# Create a dictionary to store the list of targeting perturbation IDs for each protein
targeting_pert_ids = {protein: [] for protein in df.columns[:-12]}
for index, row in df_valid_targets.iterrows():
    target_list = split_by_semi(row['Uniprot.ID'])
    for target in target_list:
        if target in targeting_pert_ids:
            targeting_pert_ids[target].append(row['pert_id'])

# Check if a protein is a target
is_target = df.columns[:-12].isin(target_proteins)

# Create the new DataFrame
result_df = pd.DataFrame({
    'NaN_prop_samples_with_targets': nan_prop_samples_with_targets,
    'NaN_prop_all_samples': nan_prop_all_samples,
    'target': is_target,
    'NaN_control': nan_control,
    'targeting_pert_ids': [targeting_pert_ids[protein] if targeting_pert_ids[protein] else np.nan for protein in df.columns[:-12]]
})

# Set the index to the protein names
# result_df.index = df.columns[:-12]
result_df.index.name = "proteins"
print(result_df.head())

result_df.to_csv('prots_info.csv')
