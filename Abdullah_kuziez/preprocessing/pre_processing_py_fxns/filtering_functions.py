import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split

#/Basic incompleteness filters
def filter_proteins_with_control(data_by_cell_line, control_data_by_cell_line, cell_lines,**kwargs):
    filtered_data,graphing_dict_before,graphing_dict_after={},{},{}
    print_flag=kwargs.pop('print_flag',False)
    graph_flag=kwargs.pop('graph_flag',False)
    graph_type=kwargs.pop('graph_type','hist')
    filter_flag=kwargs.pop('filter_flag',True)
    if not filter_flag:
        print('filtering proteins with control values is disabled')
        return data_by_cell_line
    print('filtering proteins with control values')
    for cell in cell_lines:
        control_prots=list(control_data_by_cell_line[cell].dropna(axis=1).columns)
        data_prots=list(data_by_cell_line[cell].columns)

        control_prots_intersect = list(set(control_prots).intersection(data_prots))
        # Also keep any column that starts with 'meta_' from the original dataframe
        meta_cols = [col for col in data_by_cell_line[cell].columns if col.startswith('meta_')]

        for col in meta_cols:
            if col not in control_prots_intersect:
                control_prots_intersect.append(col)

        filtered_data[cell]=data_by_cell_line[cell][control_prots_intersect]
        if print_flag:
            print(f"['Cell Line: ' {cell}]  {len(data_prots)} prots -> {len(control_prots_intersect)} prots")

        if graph_flag:
            graphing_dict_before[cell] = pd.Series(
            [1 if prot in control_prots_intersect else 0 for prot in data_by_cell_line[cell].columns],
            index=data_by_cell_line[cell].columns
            )
            graphing_dict_after[cell] = pd.Series(
            [1 if prot in control_prots_intersect else 0 for prot in filtered_data[cell].columns],
            index=filtered_data[cell].columns
            )
    if graph_flag:
        #plotting before:
        plot_grid_graphs(
            graphing_dict_before,
            cell_lines,
            title_func=lambda cl: f"{cl} protein in control(before)",
            xlabel_func=lambda _: "In control (1/0)",
            graph_type=graph_type,
            **kwargs
        )
        #plotting after
        plot_grid_graphs(
            graphing_dict_after,
            cell_lines,
            title_func=lambda cl: f"{cl} protein in control(after)",
            xlabel_func=lambda _: "In control (1/0)",
            graph_type=graph_type,
            **kwargs
        )
    return filtered_data

def filter_incomplete_proteins(data_by_cell_line, cell_lines, completeness_threshold_prot=0.8,**kwargs):
    filtered_data,graphing_dict_before,graphing_dict_after={},{},{}
    print_flag=kwargs.pop('print_flag',False)
    graph_flag=kwargs.pop('graph_flag',False)
    graph_type=kwargs.pop('graph_type','hist')
    filter_flag=kwargs.pop('filter_flag',True)
    if not filter_flag:
        print('filtering incomplete proteins is disabled')
        return data_by_cell_line

    print('filtering incomplete proteins')

    for cell_line in cell_lines:
        df = data_by_cell_line[cell_line]
        before_cols = df.shape[1]
        n_trials = df.shape[0]
        min_non_na = int(completeness_threshold_prot * n_trials)

        filtered_df = df.dropna(axis=1, thresh=min_non_na)
        filtered_data[cell_line] = filtered_df
        after_cols = filtered_df.shape[1]
        if print_flag:
            print(f"[cell line: {cell_line}]  {before_cols} cols -> {after_cols} cols")
        if graph_flag:
            graphing_dict_before[cell_line]=df.isna().sum(axis=0)/df.shape[0]
            graphing_dict_after[cell_line]=filtered_df.isna().sum(axis=0)/filtered_df.shape[0]
    print('\n\n')
    if graph_flag:
        #plot before:
        plot_grid_graphs(
            graphing_dict_before,
            cell_lines,
            title_func=lambda cl: f"{cl} protein incompleteness (before)",
            xlabel_func=lambda _: "Fraction NA",
            graph_type=graph_type,
            **kwargs
        )
        #plot after:
        plot_grid_graphs(
            graphing_dict_after,
            cell_lines,
            title_func=lambda cl: f"{cl} protein incompleteness (after)",
            xlabel_func=lambda _: "Fraction NA",
            graph_type=graph_type,
            **kwargs
        )
    return filtered_data

def filter_incomplete_experiments(data_by_cell_line, cell_lines, completeness_threshold_experiment=0.8,**kwargs):
    filtered_data,graphing_dict_before,graphing_dict_after={},{},{}
    print_flag=kwargs.pop('print_flag',False)
    graph_flag=kwargs.pop('graph_flag',False)
    graph_type=kwargs.pop('graph_type','hist')
    filter_flag=kwargs.pop('filter_flag',True)
    if not filter_flag:
        print('filtering incomplete experiments is disabled')
        return data_by_cell_line

    print('filtering incomplete experiments')


    for cell_line in cell_lines:
        df = data_by_cell_line[cell_line]
        before_rows = df.shape[0]
        min_non_na = int(completeness_threshold_experiment * df.shape[1])
        filtered_df = df.dropna(axis=0, thresh=min_non_na)
        filtered_data[cell_line] = filtered_df
        after_rows = filtered_df.shape[0]
        if print_flag:
            print(f"[{cell_line}: ]  {before_rows} experiments -> {after_rows} experiments")
        if graph_flag:
            graphing_dict_before[cell_line]=df.isna().sum(axis=1)/df.shape[1]
            graphing_dict_after[cell_line]=filtered_df.isna().sum(axis=1)/filtered_df.shape[1]

    print('\n\n')
    if graph_flag:
        #plot before:
        plot_grid_graphs(
            graphing_dict_before,
            cell_lines,
            title_func=lambda cl: f"{cl} experiment incompleteness (before)",
            xlabel_func=lambda _: "Fraction NA",
            graph_type=graph_type,
            **kwargs
        )

        #plot after:
        plot_grid_graphs(
            graphing_dict_after,
            cell_lines,
            title_func=lambda cl: f"{cl} experiment incompleteness (after)",
            xlabel_func=lambda _: "Fraction NA",
            graph_type=graph_type,
            **kwargs
        )
    return filtered_data


#high variance filters
def filter_keep_low_cv(data_by_cell_line,coeffvar_by_cell_line,cell_lines,max_cv,**kwargs):
    graphing_dict_before,graphing_dict_after,filtered_data = {},{},{}
    print_flag=kwargs.pop('print_flag',False)
    graph_flag=kwargs.pop('graph_flag',False)
    graph_type=kwargs.pop('graph_type','hist')
    filter_flag=kwargs.pop('filter_flag',True)
    if not filter_flag:
        print('filtering low coefficient of variation proteins is disabled')
        return data_by_cell_line

    print('filtering low coefficient of variation proteins')

    for cell in cell_lines:
        meta_cols=data_by_cell_line[cell].columns[data_by_cell_line[cell].columns.str.contains('meta_')]
        coeff_var=coeffvar_by_cell_line[cell]
        # Only keep numeric columns, and filter those whose values are less than max_cv
        numeric_coeff_var = coeff_var.select_dtypes(include=[float, int])
        # Filter to only columns where the value is less than max_cv
        keep_cols = numeric_coeff_var.columns[numeric_coeff_var.iloc[0] < max_cv]
        keep_cols=list(set(keep_cols).intersection(set(data_by_cell_line[cell].columns)))
        keep_cols = list(keep_cols) + list(meta_cols)
        
        filtered_data[cell] = data_by_cell_line[cell][keep_cols]

        original_num_proteins = data_by_cell_line[cell].shape[1]
        filtered_num_proteins = filtered_data[cell].shape[1]
        if print_flag:
            print(f"Cell line: {cell} | Filtered from {original_num_proteins} to {filtered_num_proteins} proteins")

        if graph_flag:
            # Get a pandas Series of the numeric coefficient of variation values before filtering
            col_set_1=data_by_cell_line[cell].select_dtypes(include=[float, int]).columns.intersection(set(numeric_coeff_var.columns))
            graphing_dict_before[cell] = numeric_coeff_var[col_set_1].iloc[0, :]
            intersect = list(set(keep_cols).intersection(set(coeff_var.columns)))
            # Get a pandas Series of the numeric coefficient of variation values after filtering
            graphing_dict_after[cell] = numeric_coeff_var[intersect].iloc[0, :]

    print('\n\n')
    if graph_flag:
        #plot before:
        plot_grid_graphs(
            graphing_dict_before,
            cell_lines,
            title_func=lambda cl: f"{cl} Coefficient of variation by protein",
            xlabel_func=lambda _: "coefficient of variation",
            graph_type=graph_type,
            **kwargs
        )

        #plot after:
        plot_grid_graphs(
            graphing_dict_after,
            cell_lines,
            title_func=lambda cl: f"{cl} Coefficient of variation by protein",
            xlabel_func=lambda _: "coefficient of variation",
            graph_type=graph_type,
            **kwargs
        )
    return filtered_data

def remove_outlier_proteins(data_by_cell_line, cell_lines, outlier_factor=10, **kwargs):
    """
    Remove protein columns from each cell line DataFrame that have mean values
    much larger than the rest (default: > outlier_factor * mean of means).
    Optionally plot before and after protein distributions for each cell line using plot_cdf_grid.
    """
    graphing_dict_before,graphing_dict_after,filtered_data = {},{},{}
    print_flag=kwargs.pop('print_flag',False)
    graph_flag=kwargs.pop('graph_flag',False)
    graph_type=kwargs.pop('graph_type','hist')
    filter_flag=kwargs.pop('filter_flag',True)
    if not filter_flag:
        print('filtering outlier proteins is disabled')
        return data_by_cell_line

    print('removing outlier proteins')

    for cell_line in cell_lines:
        data_cell = data_by_cell_line[cell_line]

        # Only consider numeric columns for outlier detection
        numeric_cols = data_cell.select_dtypes(include=[float, int]).columns
        col_means = data_cell[numeric_cols].mean(axis=0)
        mean_of_means = col_means.mean()

        # Keep columns whose mean is less than outlier_factor * mean_of_means
        keep_cols = col_means[col_means < outlier_factor * mean_of_means].index
        # Retain all non-numeric columns, plus filtered numeric columns
        non_numeric_cols = data_cell.columns.difference(numeric_cols)
        final_cols = list(non_numeric_cols) + list(keep_cols)
        data_cell_filtered = data_cell[final_cols]
        filtered_data[cell_line] = data_cell_filtered

        num_proteins_before = len(numeric_cols)
        num_proteins_after = len(keep_cols)
        if print_flag:
            print(f"{cell_line}: started with {num_proteins_before} proteins, filtered to {num_proteins_after} proteins")
        if graph_flag:

            graphing_dict_before[cell_line] = col_means
            graphing_dict_after[cell_line] = col_means[keep_cols]

    
    print('\n\n')
    if graph_flag:
        #plot before:
        plot_grid_graphs(
            graphing_dict_before,
            cell_lines,
            title_func=lambda cl: f"{cl} Outlier proteins",
            xlabel_func=lambda _: "mean expression per protein",
            graph_type=graph_type,
            **kwargs
        )

        #plot after:
        plot_grid_graphs(
            graphing_dict_after,
            cell_lines,
            title_func=lambda cl: f"{cl} Outlier proteins",
            xlabel_func=lambda _: "mean expression per protein",
            graph_type=graph_type,
            **kwargs
        )
    return filtered_data


#log transforming:
def log2_transform_by_control(data_by_cell_line, control_data_by_cell_line, cell_lines):
    """
    Log2 transform each protein column in data_by_cell_line by dividing by the corresponding control value,
    then taking the log2 of the result.
    """
    print('log2 transforming by control')
    log2_transformed = {}
    for cell in cell_lines:
        df = data_by_cell_line[cell].copy()
        control_vals = control_data_by_cell_line[cell].copy()
        control_vals['meta_Inhi_5']=100
        control_vals['meta_Inhi_05']=100
        control_vals['meta_Inhi_50']=100
        control_vals['meta_Inhi_200']=100
        
        # Only transform columns that are present in both data and control, and are numeric
        protein_cols=list(set(df.select_dtypes(include='number').columns).intersection(set(control_vals.columns)))
        for col in protein_cols:
            control_val = control_vals[col].values

            df[col] = np.log2(df[col] / control_val)

        log2_transformed[cell] = df.fillna(0)
    print('\n\n')
    return log2_transformed


#complex filters
def filter_by_mutual_information(data_by_cell_line,cell_lines, mi_thresh=0.01,y_col=None,rand_state=42,**kwargs):
    filter_flag=kwargs.pop('filter_flag',True)
    if not filter_flag:
        print('filtering by mutual information is disabled')
        return data_by_cell_line

    filtered_data,graphing_dict_before,graphing_dict_after={},{},{}
    print_flag=kwargs.pop('print_flag',False)
    graph_flag=kwargs.pop('graph_flag',False)
    graph_type=kwargs.pop('graph_type','hist')
    print('filtering by mutual information')



    for cell in cell_lines:
        raw_data=data_by_cell_line[cell]
        meta_df=raw_data.filter(regex='meta_',axis=1)
        valid_cols=raw_data.drop(columns=meta_df.columns)

        y = raw_data[y_col]
        mi = mutual_info_regression(
            valid_cols.values, 
            y.values, 
            random_state=rand_state
        )
        mi_series = pd.Series(mi, index=valid_cols.columns)
        keep_cols = list(mi_series[mi_series >= mi_thresh].index)

        if y_col in valid_cols.columns and y_col not in keep_cols:
            keep_cols.append(y_col)
        filtered_data[cell] = pd.merge(valid_cols[keep_cols],meta_df,left_index=True,right_index=True)
        if print_flag:
            print(f"[{cell}] Filtered proteins by mutual information: {raw_data.shape[1]} -> {valid_cols[keep_cols].shape[1]}")

        if graph_flag:
            graphing_dict_before[cell] = mi_series
            graphing_dict_after[cell] = mi_series[keep_cols]

    if graph_flag:
        plot_grid_graphs(
            graphing_dict_before,
            cell_lines,
            title_func=lambda cl: f"{cl} Mutual Information by protein (before)",
            xlabel_func=lambda _: "Mutual Information",
            graph_type=graph_type,
            **kwargs
        )
        plot_grid_graphs(
            graphing_dict_after,
            cell_lines,
            title_func=lambda cl: f"{cl} Mutual Information by protein (after)",
            xlabel_func=lambda _: "Mutual Information",
            graph_type=graph_type,
            **kwargs
        )
    print('\n\n')
    return filtered_data

def iterative_signal_filtering(
    data_by_cell_line,cell_lines, 
    std_threshold=2.5, 
    tole=0.001, 
    filtering_to_use=1, 
    **kwargs):
    #logic of the code: assume we are dealing with gaussian noise initially, and then if there are things that stand out from the noise, we pluck them out
    #then we repeat the process until it seems like we have plucked out all the signal and we are truly left with just noise, and then we return all the proteins
    #that have a single non-signal value
    print('filtering using iterative SNR filtering')
    print_flag=kwargs.pop('print_flag',False)
    graph_flag=kwargs.pop('graph_flag',False)
    graph_type=kwargs.pop('graph_type','hist')
    graphing_dict_before,graphing_dict_after,filtered_data={},{},{}
    filter_flag=kwargs.pop('filter_flag',True)
    if not filter_flag:
        print('filtering by iterative signal filtering is disabled')
        return data_by_cell_line


    for cell in cell_lines:
        df_expr=data_by_cell_line[cell]
        meta_df=df_expr.filter(regex='meta_',axis=1)
        df_expr=df_expr.drop(columns=meta_df.columns)
        n_start = df_expr.shape[1]

        std_old = 0.0
        std_new = 0.0
        non_signal_idxs = []
        signal_matrix = np.zeros(df_expr.shape)  # assume we are dealing with gaussian noise initially (nothing is signal to start with)
        exclusion_matrix=np.ones(df_expr.shape) #since there is no signal, we exclude nothing to start with

        while (std_old == 0 or abs(std_new - std_old) > tole):
            #this logic is the same regardless of the filtering option:
            std_old = std_new
            included_vals = np.where(exclusion_matrix == 1, df_expr.to_numpy(), np.nan) #condition, true value, false value
            std_new = np.nanstd(included_vals.flatten())
            mean_new = np.nanmean(included_vals.flatten())
            matrix_of_stds = np.abs((included_vals - mean_new) / std_new) #calculates the standard deviation for all included values
            #in method 1; if an expression value looks like it is signal, we then exclude just that value from the iterations in the future
            if(filtering_to_use==1):
                signal_matrix[matrix_of_stds > std_threshold] = 1 #all values that are greater than the std threshold are signal
                exclusion_matrix=exclusion_matrix*(1-signal_matrix) #all values that are signal are excluded
            #in method 2 if an expression value looks like it is signal, we then exclude that entire protein from future iterations
            if(filtering_to_use==2):
                signal_matrix[matrix_of_stds > std_threshold] = 1 #all values that are greater than the std threshold are signal
                #now say if there is a single signal entry in a column, that entire column is signal
                signal_cols = np.where(signal_matrix.sum(axis=0) > 0)[0]
                signal_matrix[:, signal_cols] = 1
                exclusion_matrix=exclusion_matrix*(1-signal_matrix) #all values that are signal are excluded
                
        signal_cols = np.where(signal_matrix.sum(axis=0) > 0)[0] 
        signal_col_names = df_expr.columns[signal_cols]
        filtered_df = df_expr[signal_col_names]
        filtered_data[cell] = pd.merge(filtered_df,meta_df,left_index=True,right_index=True)
        if print_flag:
            print(f"[{cell}] Filtered from {n_start} to {filtered_df.shape[1]} signal proteins.")

        if graph_flag:
            print('graphing not implemented')
            # graphing_dict_before[cell] = matrix_of_stds
            # graphing_dict_after[cell] = matrix_of_stds[signal_cols]

    if graph_flag:
        print('graphing not implemented')
        # plot_grid_graphs(
        #     graphing_dict_before,
        #     cell_lines,
        #     title_func=lambda cl: f"{cl} Standard Deviation by protein (before)",
        #     xlabel_func=lambda _: "Standard Deviation",
        #     graph_type=graph_type,
        #     **kwargs
        # )
        # plot_grid_graphs(
        #     graphing_dict_after,
        #     cell_lines,
        #     title_func=lambda cl: f"{cl} Standard Deviation by protein (after)",   
        #     xlabel_func=lambda _: "Standard Deviation",
        #     graph_type=graph_type,
        #     **kwargs
        # )
    print('\n\n')
    return filtered_data


#ablation study
def ablation_study(df,folds=5,target_column='Cell_viability%_(cck8Drug-blk)/(control-blk)*100',model='ElasticNet',n_features_to_select=100,verbosity=2):

    #generating five folds of the data:
 
    overall_data=df
    folds=np.arange(0,4)
    train_splits=[]
    test_splits=[]
    ys=[]   
    for trial in folds:
        train,test=train_test_split(overall_data,test_size=0.2,random_state=np.random.randint(1,1000000))
        ys.append(train[target_column])
        train.drop(columns=[target_column],inplace=True)
        test.drop(columns=[target_column],inplace=True)
        train_splits.append(train)
        test_splits.append(test)
    #running ablation studies on each of the five folds:
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import ElasticNetCV
    top_100_features=[]

    # #injecting noise temporarily:
    # for i,training_data in enumerate(train_splits):
    #     noisy_train = training_data + np.random.normal(0, 10, size=training_data.shape)
    #     train_splits[i] = noisy_train

    if model=='ElasticNet':
        from sklearn.linear_model import ElasticNet
        model=ElasticNet(
            alpha=0.01,
            l1_ratio=0.5,
            max_iter=500,
            random_state=42,
        )
    for trial in folds:
        rfe = RFE(estimator=model, n_features_to_select=n_features_to_select)
        rfe.fit(train_splits[trial],ys[trial])
        top_100_features.append(train_splits[trial].columns[rfe.support_])

        if (verbosity>=2):
            print(top_100_features[trial])

    # Check for consistency across splits
    for i in range(len(top_100_features)):
        for j in range(i+1, len(top_100_features)):
            overlap = set(top_100_features[i]).intersection(set(top_100_features[j]))
            if (verbosity>=1):
                print(f"Overlap between split {i} and {j}: {len(overlap)} features")
    return(top_100_features)

def overlapping_features(top_100_features_list,n_prots_to_keep=100):

        # Create a flat list of all proteins across all folds
    all_proteins = sorted(set([protein for fold in top_100_features_list for protein in fold]))

    # Create a matrix (DataFrame) where rows are proteins and columns are folds
    protein_fold_matrix = pd.DataFrame(0, index=all_proteins, columns=[f"Fold_{i}" for i in range(len(top_100_features_list))])

    # Fill the matrix: 1 if protein is present in that fold, 0 otherwise
    for i, fold in enumerate(top_100_features_list):
        protein_fold_matrix.loc[fold, f"Fold_{i}"] = 1

    # Add a column for the total count across all folds
    protein_fold_matrix['Total_Count'] = protein_fold_matrix.sum(axis=1)

    print("Protein occurrence matrix across folds:")
    print(protein_fold_matrix)

    # Optionally, show proteins sorted by how often they appear
    print("\nProteins sorted by frequency across folds:")
    protein_fold_matrix.sort_values(by='Total_Count',ascending=False,inplace=True)
    print(protein_fold_matrix['Total_Count'])
    top_100=protein_fold_matrix.index.tolist()[:n_prots_to_keep]
    return(top_100,protein_fold_matrix)


#plotting fxns
def plot_grid_graphs(data_dict,cell_lines,title_func,xlabel_func,graph_type='CDF',**kwargs):


    #setting up options for plots
    defaults = {
        "color": "#8856a7",
        "n_rows": 2,
        "n_cols": 3,
        "figsize_per_subplot": (4, 3)
    }
    config={**defaults,**kwargs}
    n_rows=config['n_rows']
    n_cols=config['n_cols']
    figsize_per_subplot=config['figsize_per_subplot']
    n_rows = config.pop('n_rows')  # extract layout option
    n_cols = config.pop('n_cols')
    figsize = config.pop('figsize_per_subplot')


    #making the figures
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize_per_subplot[0] * n_cols, figsize_per_subplot[1] * n_rows))
    axes = axes.flatten()

    for i, cell_line in enumerate(cell_lines):
        values = data_dict[cell_line]
        if graph_type=='CDF':
            sns.ecdfplot(values, ax=axes[i],**config)
            ylabel='CDF'
        if graph_type=='hist':
            sns.histplot(values, ax=axes[i],**config)
            ylabel='Count'

        axes[i].set_title(title_func(cell_line))
        axes[i].set_xlabel(xlabel_func(cell_line))
        axes[i].set_ylabel(ylabel)
    # Hide unused axes
    for j in range(len(cell_lines), n_rows * n_cols):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()
    return
#

#spearman and pearson filtering
def spearman_corr_filtering(data_by_cell_line, cell_lines, ycol='meta_Inhi_5', threshold=0.01, **kwargs):
    from scipy.stats import spearmanr
    filtered_data,graphing_dict_before,graphing_dict_after={},{},{}
    print_flag=kwargs.pop('print_flag',False)
    graph_flag=kwargs.pop('graph_flag',False)
    graph_type=kwargs.pop('graph_type','hist')
    filter_flag=kwargs.pop('filter_flag',True)
    if not filter_flag:
        print('filtering by spearman correlation is disabled')
        return data_by_cell_line
    
    print('filtering by spearman correlation')

    for cell in cell_lines:
        # Separate metadata and protein data
        meta_cols = data_by_cell_line[cell].filter(regex='meta_', axis=1)
        cell_data = data_by_cell_line[cell].drop(columns=meta_cols.columns)

        # Extract target/viability column
        if ycol not in data_by_cell_line[cell].columns:
            if print_flag:
                print(f"Warning: {ycol} not found in cell line {cell}. Skipping.")
            continue
        viability_data = data_by_cell_line[cell][ycol]

        # Calculate Spearman correlation for each protein column
        correlations = {}
        for col in cell_data.columns:
            x = cell_data[col]
            # Remove NaNs for valid correlation computation
            valid_idx = x.notna() & viability_data.notna()
            if valid_idx.sum() > 1:  # Need at least 2 valid points
                corr, _ = spearmanr(x[valid_idx], viability_data[valid_idx])
                correlations[col] = abs(corr) if not pd.isna(corr) else 0
            else:
                correlations[col] = 0

        correlations_series = pd.Series(correlations)
        
        # Filter by threshold
        keep_cols = [col for col, corr in correlations.items() if corr >= threshold]
        filtered_data[cell] = pd.merge(cell_data[keep_cols],meta_cols,left_index=True,right_index=True)

        if print_flag:
            print(f"[{cell}] Filtered proteins by spearman correlation: {cell_data.shape[1]} -> {len(keep_cols)}")

        if graph_flag:
            graphing_dict_before[cell] = correlations_series
            graphing_dict_after[cell] = correlations_series[keep_cols]

    if graph_flag:
        plot_grid_graphs(
            graphing_dict_before,
            cell_lines,
            title_func=lambda cl: f"{cl} Spearman Correlation by protein (before)",
            xlabel_func=lambda _: "Spearman Correlation",
            graph_type=graph_type,
            **kwargs
        )
        plot_grid_graphs(
            graphing_dict_after,
            cell_lines,
            title_func=lambda cl: f"{cl} Spearman Correlation by protein (after)",
            xlabel_func=lambda _: "Spearman Correlation",
            graph_type=graph_type,
            **kwargs
        )
    print('\n\n')
    return filtered_data

def pearson_corr_filtering(data_by_cell_line, cell_lines, ycol='meta_Inhi_5', threshold=0.01, **kwargs):
    from scipy.stats import pearsonr
    filtered_data,graphing_dict_before,graphing_dict_after={},{},{}
    print_flag=kwargs.pop('print_flag',False)
    graph_flag=kwargs.pop('graph_flag',False)
    graph_type=kwargs.pop('graph_type','hist')
    filter_flag=kwargs.pop('filter_flag',True)
    if not filter_flag:
        print('filtering by pearson correlation is disabled')
        return data_by_cell_line
    
    print('filtering by pearson correlation')

    for cell in cell_lines:
        # Separate metadata and protein data
        meta_cols = data_by_cell_line[cell].filter(regex='meta_', axis=1)
        cell_data = data_by_cell_line[cell].drop(columns=meta_cols.columns)

        # Extract target/viability column
        if ycol not in data_by_cell_line[cell].columns:
            if print_flag:
                print(f"Warning: {ycol} not found in cell line {cell}. Skipping.")
            continue
        viability_data = data_by_cell_line[cell][ycol]

        # Calculate Pearson correlation for each protein column
        correlations = {}
        for col in cell_data.columns:
            x = cell_data[col]
            # Remove NaNs for valid correlation computation
            valid_idx = x.notna() & viability_data.notna()
            if valid_idx.sum() > 1:  # Need at least 2 valid points
                corr, _ = pearsonr(x[valid_idx], viability_data[valid_idx])
                correlations[col] = abs(corr) if not pd.isna(corr) else 0
            else:
                correlations[col] = 0

        correlations_series = pd.Series(correlations)
        
        # Filter by threshold
        keep_cols = [col for col, corr in correlations.items() if corr >= threshold]
        filtered_data[cell] = pd.merge(cell_data[keep_cols],meta_cols,left_index=True,right_index=True)

        if print_flag:
            print(f"[{cell}] Filtered proteins by pearson correlation: {cell_data.shape[1]} -> {len(keep_cols)}")

        if graph_flag:
            graphing_dict_before[cell] = correlations_series
            graphing_dict_after[cell] = correlations_series[keep_cols]

    if graph_flag:
        plot_grid_graphs(
            graphing_dict_before,
            cell_lines,
            title_func=lambda cl: f"{cl} Pearson Correlation by protein (before)",
            xlabel_func=lambda _: "Pearson Correlation",
            graph_type=graph_type,
            **kwargs
        )
        plot_grid_graphs(
            graphing_dict_after,
            cell_lines,
            title_func=lambda cl: f"{cl} Pearson Correlation by protein (after)",
            xlabel_func=lambda _: "Pearson Correlation",
            graph_type=graph_type,
            **kwargs
        )
    print('\n\n')
    return filtered_data




