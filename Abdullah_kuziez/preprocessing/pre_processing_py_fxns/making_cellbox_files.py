import pandas as pd
import numpy as np
import re

def make_activity_nodes(targeted_proteins_with_metadata,pert_id_to_targets_dict):
    activity_nodes=targeted_proteins_with_metadata.copy()
    #scan through each protein column
    for protein in targeted_proteins_with_metadata.columns:

        #scan through each row and get the pert_id, then look it up in the dictionary to see if the protein is targeted;
        for index, row in targeted_proteins_with_metadata.iterrows():

            pert_id=row['pert_id']

            #check if the pert_id is in the pert_ID_list
            if pert_id in pert_id_to_targets_dict.keys():
                #check if the protein is in the dictionary
                if protein in pert_id_to_targets_dict[pert_id]:
                    pass
                else:
                    if protein in activity_nodes.columns:
                        activity_nodes.loc[index,protein]=0

    #cleave off last 12 columns for metadata:
    activity_nodes=activity_nodes.iloc[:,:]
    return activity_nodes

def get_targeted_indices(targeted_proteins_with_metadata):
    protein_list=targeted_proteins_with_metadata.columns
    targeted_indices = []
    #first check that it is proteins:
    for index, Uniprots in enumerate(targeted_proteins_with_metadata['Uniprot.ID']):
        if Uniprots is not np.nan:
            if 'not' not in str(Uniprots).lower():
                id_list=[]
                id_list.extend([t.strip().upper() for t in re.split(r'[;,]', Uniprots)])
                #then check that the protein is in the list of targeted proteins:
                for i in id_list:
                    if i in protein_list:
                        targeted_indices.append(index)
                        break
    return targeted_indices

def make_cellbox_files(prot_log, acti_df, file_prefix, file_path):
    #cellbox takes in three files: expr.csv, pert.csv, node_Index.csv
#expr.csv is the expression data, of size drug trials x(proteins+phenotypes+activity nodes)
#pert.csv is the perturbation data, of size drug trials x(proteins+phenotypes+activity nodes); the proteins and phenotypes are zeroed out
#the activity nodes indicate the activity of each protein in each drug trial, and so therefore should be of size #of targeted proteins, 
#these are activated.

    """
    Creates CellBox input files from processed data.
    
    Args:
        prot_log: DataFrame containing log ratios
        acti_df: DataFrame containing activity nodes
        file_prefix: Prefix for output files
        file_path: Path to save output files
    
    Returns: cellbox_files
    """

    expr_csv = prot_log.merge(acti_df, left_index=True, right_index=True)

    # Create perturbation data
    zeros_pert = pd.DataFrame(np.zeros_like(prot_log), columns=prot_log.columns, index=prot_log.index)
    acti_df_arctanh = pd.DataFrame(
        np.arctanh(acti_df.to_numpy().astype(float)),
        columns=acti_df.columns, index=acti_df.index
    )
    pert_csv = pd.merge(zeros_pert, acti_df_arctanh, left_index=True, right_index=True)

    # Create node index
    columns = pert_csv.columns.tolist()
    node_index_csv = pd.DataFrame({"A": columns})

    # Save files
    expr_csv.to_csv(
        (file_path + file_prefix + "expr.csv"),
        header=False,
        index=False
    )
    pert_csv.to_csv(
        (file_path + file_prefix + "pert.csv"),
        header=False,
        index=False
    )
    node_index_csv.to_csv(
        (file_path + file_prefix + "node_Index.csv"),
        sep=" ",
        header=False,
        index=False
    )
    return expr_csv, pert_csv, node_index_csv