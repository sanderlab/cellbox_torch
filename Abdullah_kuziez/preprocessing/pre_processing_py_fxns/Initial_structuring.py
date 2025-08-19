"""
README:
This file contains functions that shape the data into an intelligble format for the rest of the downstream analysis
in general, these functions will change for different datasets. if properly adjusted for different datasets, the 
downstream filtering and plotting functions should remain the exact same:

End structure of the data:
data_by_cell_line: a dictionary where the keys are the cell lines, and the values are dataframes with the following structure:
Rows=experiments, columns=proteins+phenotypes+metadata (noted with the regex prefix meta_!!)

"""
import pandas as pd

#splits the protein data into targeted and non-targeted proteins
#does so on a per cell line basis; looks at the set of proteins that are targeted for the given cell line based on 
#looking at the pert_id_targets_dict and seeing what pert_ids are present for the given cell line, then intersecting the proteins 
#that are targeted in those perts with the overall proteins for that cell line
def split_tgt_and_non_tgt_prots(intermediate_data,cell_lines):
    pert_dict=intermediate_data['drug_pert_id_targets_dict']
    raw_data=intermediate_data['data_by_cell_line_raw']
    tgt_prots,non_tgt_prots={},{}

    list_tgt_prots_by_cell_line=get_tgt_prots_per_cell_line(intermediate_data,cell_lines)
    

    for cell in cell_lines:
        intersection_tgt_prots=list(set(list_tgt_prots_by_cell_line[cell]).intersection(set(raw_data[cell].columns)))
        #handling meta_data
        meta_cols=raw_data[cell].columns[raw_data[cell].columns.str.contains('meta_')]
        tgt_prots[cell]=raw_data[cell][intersection_tgt_prots+list(meta_cols)]
        non_tgt_prots[cell]=raw_data[cell].drop(columns=intersection_tgt_prots) #this should be ok because it only drops the proteins


    return tgt_prots,non_tgt_prots
        
def get_tgt_prots_per_cell_line(intermediate_data,cell_lines):
    pert_dict=intermediate_data['drug_pert_id_targets_dict']
    raw_data=intermediate_data['data_by_cell_line_raw']
    list_tgt_prots_by_cell_line={}

    #tgt prots per cell line
    for cell in cell_lines:
        pert_ids=raw_data[cell]['meta_pert_id']
        #convert the pert_ids into a list of ints:
        list_pert_ids=[]
        for pert_id in pert_ids:
            pert_id=int(pert_id.replace('#', '').strip())
            list_pert_ids.append(pert_id)
        #get the targeted proteins for the cell line:
        flat_targeted_prots=[]
        for pert_id in list_pert_ids:
            targeted_prots=pert_dict[pert_id]
            flat_targeted_prots.extend(targeted_prots)
        
        list_tgt_prots_by_cell_line[cell]=flat_targeted_prots
    return list_tgt_prots_by_cell_line

#recombining data into a unified dictionary:
def recombine_data(non_tgt_prots,tgt_prots,cell_lines):
    combined_data={}
    for cell in cell_lines:
        meta_cols=tgt_prots[cell].columns[tgt_prots[cell].columns.str.contains('meta_')]
        combined_data[cell]=pd.concat([non_tgt_prots[cell].drop(columns=meta_cols),tgt_prots[cell]],axis=1)
    return combined_data













