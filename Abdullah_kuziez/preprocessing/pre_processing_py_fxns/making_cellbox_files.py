import pandas as pd
import numpy as np
import re
import os


#rewriting the make cellbox files:
def make_cellbox_files2(non_tgt_prots,phenotype_columns,tgt_prots,drug_pert_id_targets_dict,save_path,file_prefix=""):
    
    #step 0 organizing the proteins into the final format:
    meta_cols=[col for col in non_tgt_prots.columns if col.startswith('meta_')]
    len_meta=len(meta_cols)

    cleaned_non_tgt=non_tgt_prots.drop(columns=meta_cols)

    phenotype_cols=tgt_prots[phenotype_columns]
    phenotype_cols = phenotype_cols.rename(columns=lambda col: col[5:] if col.startswith('meta_') else col)

    overall_with_meta=pd.concat([cleaned_non_tgt,phenotype_cols,tgt_prots],axis=1)
    overall_without_meta=overall_with_meta.drop(columns=meta_cols)

    #step 1 make the index (without meta_cols)
    node_index_csv=make_node_index_csv(overall_without_meta)
    #step 2 make the expr csv:
    pert_csv,all_tgt_prots=make_pert_csv(overall_with_meta,drug_pert_id_targets_dict)
    #step 3 make the pert csv:
    expr_csv=make_expr_csv(overall_with_meta,pert_csv,all_tgt_prots)



    #step 4 drop all the meta_cols
    expr_csv=expr_csv.drop(columns=meta_cols)
    pert_csv=pert_csv.drop(columns=meta_cols)

    #step 5 write to files
    os.makedirs(os.path.join(save_path, file_prefix), exist_ok=True)
    node_index_csv.to_csv(
        os.path.join(save_path, file_prefix, file_prefix + 'node_Index.csv'),
        header=False,
        index=False
    )
    expr_csv.to_csv(
        os.path.join(save_path, file_prefix, file_prefix + "expr.csv"),
        header=False,
        index=False
    )
    pert_csv.to_csv(
        os.path.join(save_path, file_prefix, file_prefix + "pert.csv"),
        header=False,
        index=False
    )

    return expr_csv,pert_csv,node_index_csv


def make_pert_csv(overall_with_meta,pert_id_to_targets_dict):
    activity_nodes=overall_with_meta.copy()
    all_tgt_prots=[]
    for idx,row in activity_nodes.iterrows():
        #there is a large bug here wher I can't get into the dict because of type mismatch
        pert_id=row.loc['meta_pert_id']
        pert_id = int(str(pert_id).replace('#', '').strip())

        if pert_id in pert_id_to_targets_dict.keys():

            #get the targeted proteins and check which ones are in the column names of tgt_prots_plus_meta
            targeted_proteins=pert_id_to_targets_dict[pert_id]
            
            #get the intersection of targeted_proteins and col names
            intersection_of_prots=list(set(targeted_proteins).intersection(set(activity_nodes.columns)))
            opposite_cols=list(set(activity_nodes.columns).difference(set(intersection_of_prots)))
            activity_nodes.loc[idx,opposite_cols]=0
            all_tgt_prots.extend(intersection_of_prots)
            


    # take the arctanh of activity nodes (i.e., all_tgt_prots columns)
    all_tgt_prots=list(set(all_tgt_prots))
    import numpy as np
    if all_tgt_prots:
        activity_nodes.loc[:, all_tgt_prots] = np.arctanh(activity_nodes.loc[:, all_tgt_prots].astype(float))
        x=1
    return activity_nodes,all_tgt_prots

def make_expr_csv(all_prots_plus_meta,pert_csv,all_tgt_prots):
    
    new_expr=all_prots_plus_meta.copy()
    new_expr.loc[:,all_tgt_prots] = pert_csv.loc[:,all_tgt_prots]
    return new_expr

def make_node_index_csv(prots_with_metadata):
    node_index=[col for col in prots_with_metadata.columns if not col.startswith('meta_')]
    node_index_csv=pd.DataFrame(node_index,columns=['node_index'])
    return node_index_csv


