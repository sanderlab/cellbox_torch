import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pdb
from scipy.stats import zscore
from sklearn.impute import SimpleImputer

sns.set_theme()

df = pd.read_csv("data.csv", low_memory=False)
df["pert_id"] = df["pert_id"].astype(str)
prots_info = pd.read_csv("prots_info.csv")

def filter_by_targets():
    prots_id = df.columns[:-12]
    prots_id = [x.upper() for x in prots_id]
    targets = df["Uniprot.ID"].astype(str).tolist()
    targets = [x.replace(";", ",").upper() for x in targets]

    has_targets = []
    for t in targets:
        t_l = t.split(",")
        if len(set(t_l).intersection(set(prots_id))) > 0:
            has_targets.append(True)
        else:
            has_targets.append(False)

    df_new = df
    df_new["Uniprot.ID"] = targets
    df_new["has_targets"] = has_targets
    all_targets = []
    main_target = []
    nan_prop = []
    amount_inhibited = []

    for i, t in enumerate(targets):
        t_l = t.split(",")
        common_targets = list(set(t_l).intersection(set(prots_id)))
        if len(common_targets) > 0:
            all_targets.append(",".join(common_targets))
            p_df = prots_info[prots_info["proteins"].isin(common_targets)]

            # Remove targets which have proteomic responses higher than control
            ind = df_new["pert_id"].tolist()[i]
            ctrl = df_new[df_new["pert_id"] == 'control'][common_targets].dropna(axis=1)
            if (ctrl.shape[1] == 0):
                print(f"Warning: All targets {common_targets} in experiment pert_id={ind} have NaN in control")
                nan_prop.append(-1.0)
                main_target.append(np.nan)
                amount_inhibited.append(np.nan)
                continue

            treat = df_new[df_new["pert_id"] == ind][ctrl.columns]
            diff = pd.DataFrame(treat.to_numpy() - ctrl.to_numpy(), columns=treat.columns)
            prot_min = diff.idxmin(axis=1)
            prot_inhibited = diff[diff < 0.0].dropna(axis=1).columns.tolist()

            # Also sort targets based on their NaN props across all samples
            inhibited_targets = list(set(common_targets).intersection(prot_inhibited))
            if len(inhibited_targets) == 0:
                print(f"Warning: All targets {common_targets} in experiment pert_id={ind} are not inhibited")
                nan_prop.append(-1.0)
                main_target.append(np.nan)
                amount_inhibited.append(np.nan)
                continue
            inhibit_prot_info = prots_info[prots_info["proteins"].isin(inhibited_targets)]

            # Find main targets, satisfying two following conditions:
            # The target needs to have the lowest NaN proportion across samples
            # If there are multiple, then the target being mostly inhibited will be chosen
            min_nan_prot = inhibit_prot_info["NaN_prop_all_samples"].min()
            main_target_prot_info = inhibit_prot_info[inhibit_prot_info["NaN_prop_all_samples"] == min_nan_prot]

            if main_target_prot_info.shape[0] > 1:
                main_target_prot = diff[main_target_prot_info["proteins"].tolist()].idxmin(axis=1).item()
            else:
                main_target_prot = main_target_prot_info["proteins"].tolist()[0]
            #main_target_prot = inhibit_prot_info.loc[inhibit_prot_info["NaN_prop_all_samples"].idxmin()]["proteins"]
            main_target.append(main_target_prot)
            nan_prop.append(inhibit_prot_info.loc[inhibit_prot_info["NaN_prop_all_samples"].idxmin()]["NaN_prop_all_samples"])
            amount_inhibited.append(diff[main_target_prot].item())
        else:
            nan_prop.append(-1.0)
            all_targets.append(np.nan)
            main_target.append(np.nan)
            amount_inhibited.append(np.nan)

    df_new["all_targets"] = all_targets
    df_new["Main Target UniProtID"] = main_target
    df_new["nan_prop_of_target"] = nan_prop
    df_new["main_target_inhibited_amount"] = amount_inhibited

    return df_new

def filter_by_nan(df):
    df["pert_id"] = df["pert_id"].astype(str)
    nan_thres = 0.1

    # Some preprocessing
    df_tar = df[(~df["Main Target UniProtID"].isna()) | (df["pert_id"] == "control")]
    cols = df_tar.columns.tolist()[1:-17] + ["pert_id", "Cell_viability%_(cck8Drug-blk)/(control-blk)*100", "Main Target UniProtID"]
    df_tar = df_tar[cols].set_index("pert_id")

    # Remove any proteins of which more than 5% samples have NaN values.
    prots_less_than_some_nan = prots_info[prots_info["NaN_prop_all_samples"] < nan_thres]["proteins"].tolist()

    # Remove any proteins that have NaN in control
    prots_control_not_nan = prots_info[~prots_info["NaN_control"]]["proteins"].tolist()

    # Retain the proteins
    prots_retained = list(
        set(prots_less_than_some_nan)
        .intersection(set(prots_control_not_nan))
    )

    # Further extract rows that no longer have the targets after removing proteins
    main_targets = df_tar["Main Target UniProtID"].tolist()
    main_targets_retained = list(set(main_targets).intersection(set(prots_retained)))
    main_targets_rejected = list(set(main_targets) - set(prots_retained))
    return df_tar, main_targets_retained, prots_retained

def get_log_ratios(main_targets_retained, prots_retained, df_tar):
    control = df_tar.loc["control"].to_frame().T
    df_tar = df_tar[df_tar["Main Target UniProtID"].isin(main_targets_retained)]
    df_tar = pd.concat([df_tar, control])
    pert_id_to_targets = {pert: t for pert, t in zip(df_tar.index.tolist(), df_tar["Main Target UniProtID"].tolist()) if pert != "control"}
    df_tar = df_tar[prots_retained + ["Cell_viability%_(cck8Drug-blk)/(control-blk)*100"]]

    # SimpleImputer with mean without control sample
    df_tar_no_control = df_tar[df_tar.index != "control"]
    control = df_tar.loc[["control"], :]
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    np_tar = imp.fit_transform(df_tar_no_control)
    df_tar_no_control = pd.DataFrame(np_tar, columns=df_tar_no_control.columns, index=df_tar_no_control.index)
    df_tar = pd.concat([df_tar_no_control, control], axis=0)

    # Divide the control to all other samples, then take the log
    control = df_tar.loc[["control"]]
    prot = df_tar[df_tar.index != 'control']

    # Transform the data by tanh(pert/control)
    # prot_prot_only = prot.drop("Cell_viability%_(cck8Drug-blk)/(control-blk)*100", axis=1)
    # control_prot_only = control.drop("Cell_viability%_(cck8Drug-blk)/(control-blk)*100", axis=1)

    # # The prots
    # prot_only_diff = pd.DataFrame(
    #     np.tanh((prot_prot_only.to_numpy() - control_prot_only.to_numpy()).astype(float)),
    #     columns=prot_prot_only.columns,
    #     index=prot_prot_only.index
    # )

    # # The cell viability
    # prot_viab_only = prot[["Cell_viability%_(cck8Drug-blk)/(control-blk)*100"]]
    # control_viab_only = control[["Cell_viability%_(cck8Drug-blk)/(control-blk)*100"]]
    # control_only_diff = pd.DataFrame(
    #     np.tanh((prot_viab_only.to_numpy() - control_viab_only.to_numpy()).astype(float)),
    #     columns=prot_viab_only.columns,
    #     index=prot_viab_only.index
    # )
    prot_log = pd.DataFrame(
        np.log2((prot.to_numpy().astype(float)/control.to_numpy()).astype(float)),
        columns = prot.columns,
        index = prot.index
    )

    # prot_log = pd.merge(prot_only_diff, control_only_diff, left_index=True, right_index=True)
    main_targets = df[df["pert_id"].isin(df_tar.index.tolist())]["Main Target UniProtID"].tolist()
    main_targets.remove(np.nan)
    # prot_log["main_targets"] = main_targets
    return prot_log, pert_id_to_targets

def make_activity_nodes(prot_log, pert_id_to_targets):
    vec_list = []
    for pert_id, target in pert_id_to_targets.items():

        # Identify its log fold change in the corresponding pert sample
        target_pert = prot_log.loc[pert_id, target]

        # Convert it into a one-hot representation
        vec = [0.0 if t != target else target_pert for t in list(pert_id_to_targets.values())]
        if (sum(vec) == 0): print(f"{pert_id} has a problem")
        vec = [pert_id] + vec

        # Append it to a numpy array matrix
        vec_list.append(vec)

    # Convert the vec_list into a numpy array then pandas
    acti_targets = [f"a{t}" for t in list(pert_id_to_targets.values())]
    vec_np = np.array(vec_list)
    acti_df = pd.DataFrame(vec_np, columns=["pert_id"]+acti_targets).set_index("pert_id")

    # Remove columns with similar names
    acti_df = acti_df.T.drop_duplicates().T
    return acti_df

def make_cellbox_files(prot_log, acti_df):
    expr_csv = prot_log.merge(acti_df, left_index=True, right_index=True)

    # Create a dataframe similar to prot_log but filled with all 0s
    zeros_pert = pd.DataFrame(np.zeros_like(prot_log), columns=prot_log.columns, index=prot_log.index)
    acti_df_arctanh = pd.DataFrame(
        np.arctanh(acti_df.to_numpy().astype(float)),
        columns=acti_df.columns, index=acti_df.index
    )

    # Merge and save
    pert_csv = pd.merge(zeros_pert, acti_df_arctanh, left_index=True, right_index=True)
    columns = pert_csv.columns.tolist()
    node_index_csv = pd.DataFrame({"A": columns})

    expr_csv.to_csv(
        f"expr.csv",
        index=True
    )
    pert_csv.to_csv(
        f"pert.csv",
        index=True
    )
    node_index_csv.to_csv(
        f"node_Index.csv",
        sep=" ",
        header=False,
        index=False
    )
    return expr_csv, pert_csv, node_index_csv


# # Signal-to-noise code, based on two assumptions:
# # - Only data samples passing the criteria are excluded, not the whole protein column of which they are in
# # - Mean and std are calculated using the full matrix
def signal_to_noise_filter(expr_csv):
    std_old = 0.0
    std_new = 0.0
    df_expr = expr_csv.iloc[:, :-21]
    inds = []
    tole = 0.001
    signal = np.zeros(df_expr.shape)
    excluded = np.ones(df_expr.shape)

    while (std_old == 0) or (abs(std_new - std_old) > tole):
        print(f"std_old: {std_old:.3f}, std_new: {std_new:.3f}")
        std_old = std_new
        vec = df_expr.to_numpy().flatten()

        # Get the samples that are smaller than 2.5*std
        mean = np.mean(vec[signal.flatten() == 0])
        std_new = np.std(vec[signal.flatten() == 0])
        print(f"mean: {mean:.3f}, std: {std_new:.3f}")

        # Get the samples that are larger than 2.5*std
        # 1 in signal means the sample > 2.5*std and vice versa
        signal = (np.abs(df_expr.to_numpy()) > 2.5*std_new + mean)*excluded
        excluded = excluded * (1-signal)
        signal_df = pd.DataFrame(signal, columns=df_expr.columns, index=df_expr.index)
        inds.extend(signal_df.any(axis=0)[signal_df.any(axis=0)].index.tolist())

    # print(f"std_old: {std_old:.3f}, std_new: {std_new:.3f}")
    inds_method_1 = list(set(inds))




    # # Signal-to-noise code, based on two assumptions:
    # # - When a data sample is excluded, the whole protein column in which that protein stays in are excluded
    # # - Mean and std are calculated using the full matrix

    std_old = 0.0
    std_new = 0.0
    df_expr = expr_csv.iloc[:, :-21]
    inds = []
    tole = 0.001
    signal = np.zeros(df_expr.shape)
    excluded = np.ones(df_expr.shape)

    while (std_old == 0) or (abs(std_new - std_old) > tole):
        print(f"std_old: {std_old:.3f}, std_new: {std_new:.3f}")
        std_old = std_new
        vec = df_expr.to_numpy().flatten()

        # Get the samples that are smaller than 2.5*std
        mean = np.mean(vec[signal.flatten() == 0])
        std_new = np.std(vec[signal.flatten() == 0])
        print(f"mean: {mean:.3f}, std: {std_new:.3f}")

        # Get the samples that are larger than 2.5*std
        # 1 in signal means the sample > 2.5*std and vice versa
        signal = (np.abs(df_expr.to_numpy()) > 2.5*std_new + mean)*excluded
        signal_df = pd.DataFrame(signal, columns=df_expr.columns, index=df_expr.index)

        # ind is a list of indices of columns that have at least one excluded data sample
        ind = signal_df.any(axis=0)[signal_df.any(axis=0)].index.tolist()
        inds.extend(ind)
        int_ind = [signal_df.columns.tolist().index(i) for i in ind]
        excluded[:, int_ind] = np.zeros((excluded.shape[0], len(int_ind)))

    # print(f"std_old: {std_old:.3f}, std_new: {std_new:.3f}")
    inds_method_2 = list(set(inds))



    # # Signal-to-noise code, based on two assumptions:
    # # - Only data samples passing the criteria are excluded, not the whole protein column of which they are in
    # # - Mean and std are calculated for each perturbation condition

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
        print(f"Iteration {i}")
        #print(f"std_old: {std_old:.3f}, std_new: {std_new:.3f}")
        std_old = std_new
        vec = df_expr.to_numpy()

        # Get the samples that are smaller than 2.5*std
        mask = vec*excluded
        mask[mask == 0.0] = np.nan
        if np.all(np.any(np.isnan(mask), axis=0)):
            print("Break")
            break
        mean = np.nanmean(mask, axis=1, keepdims=True)
        std_new = np.nanstd(mask, axis=1, keepdims=True)
        #print(f"mean: {mean:.3f}, std: {std_new:.3f}")

        # Get the samples that are larger than 2.5*std
        # 1 in signal means the sample > 2.5*std and vice versa
        signal = (np.abs(df_expr.to_numpy()) > 2.5*std_new + mean)*excluded
        excluded = excluded * (1-signal)
        signal_df = pd.DataFrame(signal, columns=df_expr.columns, index=df_expr.index)
        inds.extend(signal_df.any(axis=0)[signal_df.any(axis=0)].index.tolist())
        # print(f"Length of inds: {len(set(inds))}")

    # #print(f"std_old: {std_old:.3f}, std_new: {std_new:.3f}")
    inds_method_3 = list(set(inds))

    # # To retain the highest fidelity proteins, we select the ones in the middle
    inds_final = list(set(inds_method_1).intersection(set(inds_method_2)).intersection(set(inds_method_3)))
    return inds_final

def variability_intensity_filter(expr_csv, df_tar):
    #dispersion_thres = 0.035
    intensity_upper = 2.0
    intensity_lower = -4.0

    # Choose target proteins
    prots_tar = [p[1:] for p in expr_csv.columns.tolist()[-20:]]

    # Choose proteins in the dataset with high log2 variance, using df_tar values (log2(intensity))
    df_tar_temp = df_tar.iloc[:, :-1].dropna(axis=1)
    variances = df_tar_temp.std(axis=0).to_frame().rename(columns={0: "std"})
    means = df_tar_temp.mean(axis=0).to_frame().rename(columns={0: "mean"})
    #variances = expr_csv.iloc[:, :-21].std(axis=0).to_frame().rename(columns={0: "std"})
    #means = expr_csv.iloc[:, :-21].mean(axis=0).to_frame().rename(columns={0: "mean"})
    mean_var_df = variances.merge(means, right_index=True, left_index=True)
    mean_var_df['dispersion'] = abs(mean_var_df["std"] / mean_var_df["mean"])

    dispersion_lower = np.percentile(mean_var_df["dispersion"], 95)
    # dispersion_upper = np.percentile(mean_var_df["dispersion"], 92)
    # prots_high_var = mean_var_df[(mean_var_df["dispersion"] >= dispersion_lower) & (mean_var_df["dispersion"] <= dispersion_upper)].index.tolist()
    prots_high_var = mean_var_df[(mean_var_df["dispersion"] >= dispersion_lower)].index.tolist()

    # Choose proteins in the dataset with no NaNs
    prots_no_nan = prots_info[prots_info["NaN_prop_all_samples"] == 0.0]["proteins"].tolist()

    # Choose proteins in the dataset with intensity in range
    df_temp = expr_csv.iloc[:, :-21]
    prots_intermediate_intensity = df_temp.loc[:, ((df_temp >= intensity_lower) & (df_temp <= intensity_upper)).all()].columns.tolist()
    prots_total = prots_tar + list(set(prots_high_var).intersection(prots_no_nan).intersection(prots_intermediate_intensity))
    # Total prots
    prots_total = list(
        set(prots_total)
        .intersection(expr_csv.columns.tolist())
    )
    print(f"Total proteins in subsetted dataset: {len(prots_total)}")
    return prots_total, prots_tar

def combine_and_save(total_proteins, expr_csv, pert_csv):
    # Subset one more time between these prots and the prots after signal-to-noise
    # prots_total = prots_tar + list(set(prots_total).intersection(inds_final))
    # print(f"Total proteins after intersection with signal-to-noise proteins: {len(prots_total)}")

    # Subset
    cell_viab_acti_cols = [a for a in expr_csv.columns.tolist() if a.startswith("a")] + ["Cell_viability%_(cck8Drug-blk)/(control-blk)*100"]
    all_cols = total_proteins + cell_viab_acti_cols
    expr_csv_sub = expr_csv[all_cols].astype(float)
    pert_csv_sub = pert_csv[all_cols].astype(float)

    # Save
    expr_csv_sub.to_csv('expr.csv', index=False, header=False)
    pert_csv_sub.to_csv('pert.csv', index=False, header=False)
    columns = pert_csv_sub.columns.tolist()
    node_index_csv = pd.DataFrame({"A": columns})
    node_index_csv.to_csv('node_Index.csv', index=False, header=False)


df_new = filter_by_targets()
df_tar, main_targets_retained, prots_retained = filter_by_nan(df)
prot_log, pert_id_to_targets = get_log_ratios(main_targets_retained, prots_retained, df_tar)
acti_df = make_activity_nodes(prot_log, pert_id_to_targets)
expr_csv, pert_csv, node_index_csv = make_cellbox_files(prot_log, acti_df)
signal_indices = signal_to_noise_filter(expr_csv)
prots_total, target_prots = variability_intensity_filter(expr_csv, df_tar)
print(len(prots_total))
total_proteins = list(set(prots_total).intersection(set(signal_indices)).union(set(target_prots)))
print(len(total_proteins))
combine_and_save(total_proteins, expr_csv, pert_csv)
