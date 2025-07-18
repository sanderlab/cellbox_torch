"""
Main preprocessing script for protein expression data analysis.
This script processes protein expression data, filters targets, handles missing values,
and prepares data for downstream analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pdb
from scipy.stats import zscore
from sklearn.impute import SimpleImputer
import argparse
from config import Config

def filter_by_targets(prot_data, prot_info, first_prot_index, last_prot_index, id_key):
    """
    Filters and processes protein targets from the dataset.
    
    Args:
        prot_data: DataFrame containing protein expression data
        prot_info: DataFrame containing protein metadata
        first_prot_index: Index of first protein column
        last_prot_index: Index of last protein column
        id_key: Column name containing target IDs
    
    Returns:
        DataFrame with processed target information
    """
    # Extract protein IDs and convert to uppercase
    prots_id = prot_data.columns[first_prot_index:last_prot_index]
    prots_id = [x.upper() for x in prots_id]

    # Process target IDs: replace semicolons with commas and convert to uppercase
    targets = prot_data[id_key].astype(str).tolist()
    targets = [x.replace(";", ",").upper() for x in targets]

    # Check which samples have valid targets in the measured proteins
    has_targets = []
    for t in targets:
        t_l = t.split(",")
        if len(set(t_l).intersection(set(prots_id))) > 0:
            has_targets.append(True)
        else:
            has_targets.append(False)

    # Add target information to the dataset
    prot_data["Uniprot.ID"] = targets
    prot_data["has_targets"] = has_targets

    # Process each target to find inhibited proteins
    all_targets = []
    main_target = []
    nan_prop = []
    amount_inhibited = []
    for i, t in enumerate(targets):
        t_l = t.split(",")
        common_targets = list(set(t_l).intersection(set(prots_id)))

        if len(common_targets) > 0:
            all_targets.append(",".join(common_targets))
            p_df = prot_info[prot_info["proteins"].isin(common_targets)]

            # Get control and treatment data for comparison
            ind = prot_data["pert_id"].tolist()[i]
            ctrl = prot_data[prot_data["pert_id"] == 'control'][common_targets].dropna(axis=1)

            # Handle cases where control data is missing
            if (ctrl.shape[1] == 0):
                print(f"Warning: All targets {common_targets} in experiment pert_id={ind} have NaN in control")
                nan_prop.append(-1.0)
                main_target.append(np.nan)
                amount_inhibited.append(np.nan)
                continue

            # Calculate protein inhibition
            treat = prot_data[prot_data["pert_id"] == ind][ctrl.columns]
            diff = pd.DataFrame(treat.to_numpy() - ctrl.to_numpy(), columns=treat.columns)
            prot_min = diff.idxmin(axis=1)
            prot_inhibited = diff[diff < 0.0].dropna(axis=1).columns.tolist()
            inhibited_targets = list(set(common_targets).intersection(prot_inhibited))

            # Handle cases where no targets are inhibited
            if len(inhibited_targets) == 0:
                print(f"Warning: All targets {common_targets} in experiment pert_id={ind} are not inhibited")
                nan_prop.append(-1.0)
                main_target.append(np.nan)
                amount_inhibited.append(np.nan)
                continue

            # Find main target based on NaN proportion and inhibition amount
            inhibit_prot_info = prot_info[prot_info["proteins"].isin(inhibited_targets)]
            min_nan_prot = inhibit_prot_info["NaN_prop_all_samples"].min()
            main_target_prot_info = inhibit_prot_info[inhibit_prot_info["NaN_prop_all_samples"] == min_nan_prot]

            if main_target_prot_info.shape[0] > 1:
                main_target_prot = diff[main_target_prot_info["proteins"].tolist()].idxmin(axis=1).item()
            else:
                main_target_prot = main_target_prot_info["proteins"].tolist()[0]

            main_target.append(main_target_prot)
            nan_prop.append(inhibit_prot_info.loc[inhibit_prot_info["NaN_prop_all_samples"].idxmin()]["NaN_prop_all_samples"])
            amount_inhibited.append(diff[main_target_prot].item())

        else:
            nan_prop.append(-1.0)
            all_targets.append(np.nan)
            main_target.append(np.nan)
            amount_inhibited.append(np.nan)

    # Add processed target information to the dataset
    prot_data["all_targets"] = all_targets
    prot_data["Main Target UniProtID"] = main_target
    prot_data["nan_prop_of_target"] = nan_prop
    prot_data["main_target_inhibited_amount"] = amount_inhibited
    return prot_data

def filter_by_nan(prot_data, prot_info, nan_thres, first_prot_index, last_prot_index, metadata_cols, id_key):
    """
    Filters proteins based on NaN proportions and control sample quality.
    
    Args:
        prot_data: DataFrame containing protein expression data
        prot_info: DataFrame containing protein metadata
        nan_thres: Threshold for maximum allowed NaN proportion
        first_prot_index: Index of first protein column
        last_prot_index: Index of last protein column
        metadata_cols: List of metadata column names
        id_key: Column name containing target IDs
    
    Returns:
        Tuple of (filtered protein data, retained main targets, retained proteins)
    """
    # Filter for drugs with targets and control samples
    prot_tar = prot_data[(~prot_data[id_key].isna()) | (prot_data["pert_id"] == "control")]
    cols = prot_tar.columns.tolist()[first_prot_index:-17] + metadata_cols
    prot_tar = prot_tar[cols]
    prot_tar = prot_tar.set_index('pert_id')

    # Filter proteins based on NaN proportions
    prots_less_than_some_nan = prot_info[prot_info["NaN_prop_all_samples"] < nan_thres]["proteins"].tolist()
    prots_control_not_nan = prot_info[~prot_info["NaN_control"]]["proteins"].tolist()

    # Get intersection of proteins meeting both criteria
    prots_retained = list(
        set(prots_less_than_some_nan)
        .intersection(set(prots_control_not_nan))
    )

    # Filter main targets
    main_targets = prot_tar["Main Target UniProtID"].tolist()
    main_targets_retained = list(set(main_targets).intersection(set(prots_retained)))
    return prot_tar, main_targets_retained, prots_retained

def get_log_ratios(prot_data, main_targets_retained, prots_retained, prot_tar, target_col, id_key):
    """
    Calculates log ratios of protein expression between treatment and control samples.
    
    Args:
        prot_data: DataFrame containing protein expression data
        main_targets_retained: List of retained main targets
        prots_retained: List of retained proteins
        prot_tar: DataFrame containing target information
        target_col: Column name containing target information
        id_key: Column name containing target IDs
    
    Returns:
        Tuple of (log ratios DataFrame, mapping of perturbation IDs to targets)
    """
    # Get control data and create perturbation ID to target mapping
    control = prot_tar.loc["control"].to_frame().T
    prot_tar = prot_tar[prot_tar['Main Target UniProtID'].isin(main_targets_retained)]
    prot_tar = pd.concat([prot_tar, control])
    pert_id_to_targets = {int(pert): t for pert, t in zip(prot_tar.index.tolist(), prot_tar['Main Target UniProtID'].tolist()) if pert != "control"}
    prot_tar = prot_tar[prots_retained + [target_col]]

    # Impute missing values using mean
    prot_tar_no_control = prot_tar[prot_tar.index != "control"]
    control = prot_tar.loc[["control"], :]
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    np_tar = imp.fit_transform(prot_tar_no_control)
    prot_tar_no_control = pd.DataFrame(np_tar, columns=prot_tar_no_control.columns, index=prot_tar_no_control.index)
    prot_tar = pd.concat([prot_tar_no_control, control], axis=0)

    # Calculate log ratios
    control = prot_tar.loc[["control"]]
    prot = prot_tar[prot_tar.index != 'control']

    prot_log = pd.DataFrame(
        np.log2((prot.to_numpy().astype(float)/control.to_numpy()).astype(float)),
        columns = prot.columns,
        index = prot.index.astype(int)
    )

    main_targets = prot_data[prot_data["pert_id"].isin(prot_tar.index.tolist())]['Main Target UniProtID'].tolist()
    main_targets.remove(np.nan)

    return prot_log, pert_id_to_targets

def get_activity_nodes(prot_log, pert_id_to_targets):
    """
    Creates activity nodes for each target protein.
    
    Args:
        prot_log: DataFrame containing log ratios
        pert_id_to_targets: Dictionary mapping perturbation IDs to targets
    
    Returns:
        DataFrame containing activity nodes
    """
    vec_list = []
    for pert_id, target in pert_id_to_targets.items():
        # Get log fold change for target
        target_pert = prot_log.loc[pert_id, target]

        # Create one-hot representation
        vec = [0.0 if t != target else target_pert for t in list(pert_id_to_targets.values())]
        if (sum(vec) == 0): print(f"{pert_id} has a problem")
        vec = [pert_id] + vec
        vec_list.append(vec)

    # Convert to DataFrame
    acti_targets = [f"a{t}" for t in list(pert_id_to_targets.values())]
    vec_np = np.array(vec_list)
    acti_df = pd.DataFrame(vec_np, columns=["pert_id"]+acti_targets).set_index("pert_id")

    # Remove duplicate columns
    acti_df = acti_df.T.drop_duplicates().T
    return acti_df

def make_cellbox_files(prot_log, acti_df, file_prefix, file_path):
    """
    Creates CellBox input files from processed data.
    
    Args:
        prot_log: DataFrame containing log ratios
        acti_df: DataFrame containing activity nodes
        file_prefix: Prefix for output files
        file_path: Path to save output files
    
    Returns:
        Tuple of (expression DataFrame, perturbation DataFrame, node index DataFrame)
    """
    # Create expression data
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
        index=True
    )
    pert_csv.to_csv(
        (file_path + file_prefix + "pert.csv"),
        index=True
    )
    node_index_csv.to_csv(
        (file_path + file_prefix + "node_Index.csv"),
        sep=" ",
        header=False,
        index=False
    )
    return expr_csv, pert_csv, node_index_csv

def signal_to_noise_filter(expr_csv):
    """
    Filters proteins based on signal-to-noise ratio using three different methods.
    
    Args:
        expr_csv: DataFrame containing expression data
    
    Returns:
        List of proteins that pass all filtering methods
    """
    # Method 1: Global signal-to-noise filtering
    std_old = 0.0
    std_new = 0.0
    df_expr = expr_csv.iloc[:, :-21]
    inds = []
    tole = 0.001
    signal = np.zeros(df_expr.shape)
    excluded = np.ones(df_expr.shape)

    while (std_old == 0) or (abs(std_new - std_old) > tole):
        std_old = std_new
        vec = df_expr.to_numpy().flatten()
        mean = np.mean(vec[signal.flatten() == 0])
        std_new = np.std(vec[signal.flatten() == 0])
        print(f"mean: {mean:.3f}, std: {std_new:.3f}")
        signal = (np.abs(df_expr.to_numpy()) > 2.5*std_new + mean)*excluded
        excluded = excluded * (1-signal)
        signal_df = pd.DataFrame(signal, columns=df_expr.columns, index=df_expr.index)
        inds.extend(signal_df.any(axis=0)[signal_df.any(axis=0)].index.tolist())

    inds_method_1 = list(set(inds))

    # Method 2: Column-based signal-to-noise filtering
    std_old = 0.0
    std_new = 0.0
    df_expr = expr_csv.iloc[:, :-21]
    inds = []
    tole = 0.001
    signal = np.zeros(df_expr.shape)
    excluded = np.ones(df_expr.shape)

    while (std_old == 0) or (abs(std_new - std_old) > tole):
        std_old = std_new
        vec = df_expr.to_numpy().flatten()
        mean = np.mean(vec[signal.flatten() == 0])
        std_new = np.std(vec[signal.flatten() == 0])
        signal = (np.abs(df_expr.to_numpy()) > 2.5*std_new + mean)*excluded
        signal_df = pd.DataFrame(signal, columns=df_expr.columns, index=df_expr.index)
        ind = signal_df.any(axis=0)[signal_df.any(axis=0)].index.tolist()
        inds.extend(ind)
        int_ind = [signal_df.columns.tolist().index(i) for i in ind]
        excluded[:, int_ind] = np.zeros((excluded.shape[0], len(int_ind)))

    inds_method_2 = list(set(inds))

    # Method 3: Per-perturbation signal-to-noise filtering
    df_expr = expr_csv.iloc[:, :-21]
    std_old = np.zeros(df_expr.shape)
    std_new = np.zeros(df_expr.shape)
    inds = []
    tole = 0.001
    signal = np.zeros(df_expr.shape)
    excluded = np.ones(df_expr.shape)
    i = -1

    while np.all(std_old == 0) or np.all(np.abs(std_new - std_old) > tole):
        i += 1
        std_old = std_new
        vec = df_expr.to_numpy()
        mask = vec*excluded
        mask[mask == 0.0] = np.nan
        if np.all(np.any(np.isnan(mask), axis=0)):
            print("Break")
            break
        mean = np.nanmean(mask, axis=1, keepdims=True)
        std_new = np.nanstd(mask, axis=1, keepdims=True)
        signal = (np.abs(df_expr.to_numpy()) > 2.5*std_new + mean)*excluded
        excluded = excluded * (1-signal)
        signal_df = pd.DataFrame(signal, columns=df_expr.columns, index=df_expr.index)
        inds.extend(signal_df.any(axis=0)[signal_df.any(axis=0)].index.tolist())

    inds_method_3 = list(set(inds))

    # Combine results from all methods
    inds_final = list(set(inds_method_1).intersection(set(inds_method_2)).intersection(set(inds_method_3)))
    return inds_final

def variability_intensity_filter(expr_csv, prot_tar, prot_info, intensity_upper, intensity_lower, dispersion_lower):
    """
    Filters proteins based on variability and intensity thresholds.
    
    Args:
        expr_csv: DataFrame containing expression data
        prot_tar: DataFrame containing target information
        prot_info: DataFrame containing protein metadata
        intensity_upper: Upper threshold for protein intensity
        intensity_lower: Lower threshold for protein intensity
        dispersion_lower: Lower threshold for protein dispersion
    
    Returns:
        Tuple of (list of total proteins, list of target proteins)
    """
    # Get target proteins
    prots_tar = [p[1:] for p in expr_csv.columns.tolist()[-19:]]

    # Calculate protein statistics
    prot_tar_temp = prot_tar.iloc[:, :-1].dropna(axis=1)
    variances = prot_tar_temp.std(axis=0).to_frame().rename(columns={0: "std"})
    means = prot_tar_temp.mean(axis=0).to_frame().rename(columns={0: "mean"})
    mean_var_df = variances.merge(means, right_index=True, left_index=True)
    mean_var_df['dispersion'] = abs(mean_var_df["std"] / mean_var_df["mean"])

    # Filter proteins based on dispersion
    dispersion_bound = np.percentile(mean_var_df["dispersion"], dispersion_lower)
    prots_high_var = mean_var_df[(mean_var_df["dispersion"] >= dispersion_bound)].index.tolist()

    # Filter proteins with no NaNs
    prots_no_nan = prot_info[prot_info["NaN_prop_all_samples"] == 0.0]["proteins"].tolist()

    # Filter proteins based on intensity range
    df_temp = expr_csv.iloc[:, :-21]
    prots_intermediate_intensity = df_temp.loc[:, ((df_temp >= intensity_lower) & (df_temp <= intensity_upper)).all()].columns.tolist()
    
    # Combine all filters
    prots_total = prots_tar + list(set(prots_high_var).intersection(prots_no_nan).intersection(prots_intermediate_intensity))
    prots_total = list(
        set(prots_total)
        .intersection(expr_csv.columns.tolist())
    )
    print(f"Total proteins in subsetted dataset: {len(prots_total)}")
    return prots_total, prots_tar

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Preprocessing Script')
    parser.add_argument('-config', '--config_path', required=True, type=str, help="Path of config")
    master_args = parser.parse_args()

    # Load configuration and data
    cfg = Config(master_args.config_path)
    prot_data = pd.read_csv(cfg.prot_data_file)
    prot_info = pd.read_csv(cfg.prot_info_file)

    # Process data through pipeline
    prot_data = filter_by_targets(prot_data, prot_info, cfg.first_prot_index, cfg.last_prot_index, cfg.id_key)
    prot_tar, main_targets_retained, prots_retained = filter_by_nan(prot_data, prot_info, cfg.nan_thres, cfg.first_prot_index, cfg.last_prot_index, cfg.metadata_cols, cfg.id_key)
    prot_log, pert_id_to_targets = get_log_ratios(prot_data, main_targets_retained, prots_retained, prot_tar, cfg.target_col, cfg.id_key)
    acti_df = get_activity_nodes(prot_log, pert_id_to_targets)
    expr_csv, pert_csv, node_index_csv = make_cellbox_files(prot_log, acti_df, cfg.file_prefix, cfg.file_path)
    inds_final = signal_to_noise_filter(expr_csv)
    prots_total, prots_tar = variability_intensity_filter(expr_csv, prot_tar, prot_info, cfg.intensity_upper, cfg.intensity_lower, cfg.dispersion_lower)

    # Combine all filters and prepare final dataset
    total_proteins = list(set(prots_total).intersection(set(inds_final)).union(set(prots_tar)))
    cell_viab_acti_cols = [a for a in expr_csv.columns.tolist() if a.startswith("a")] + ["Cell_viability%_(cck8Drug-blk)/(control-blk)*100"]
    all_cols = total_proteins + cell_viab_acti_cols
    pdb.set_trace()
    expr_csv_sub = expr_csv[all_cols].astype(float)
    pert_csv_sub = pert_csv[all_cols].astype(float)

    # Save final processed files
    expr_csv_sub.to_csv(cfg.file_path + cfg.file_prefix + 'expr.csv', index=False, header=False)
    pert_csv_sub.to_csv(cfg.file_path + cfg.file_prefix + 'pert.csv', index=False, header=False)
    columns = pert_csv_sub.columns.tolist()
    node_index_csv = pd.DataFrame({"A": columns})
    node_index_csv.to_csv(cfg.file_path + cfg.file_prefix + 'node_Index.csv', index=False, header=False)
