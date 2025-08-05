import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import LeaveOneOut
import numpy as np

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


def loo_regression_per_cell_line(
    data_by_cell_line,
    cell_lines,
    ycol,
    model,
    print_stats=True,
    plot_results=True,
    meta_regex='meta_',
    **kwargs
):
    """
    Perform Leave-One-Out regression for each cell line using the specified model.

    Parameters:
    - data_dict: dict of {cell_line: DataFrame}
    - ycol: str, name of the target column (must be present in each DataFrame)
    - model: sklearn-like estimator instance (already initialized, e.g., ElasticNet(...))
    - print_stats: bool, whether to print R2/MSE per cell line
    - plot_results: bool, whether to plot true vs predicted for each cell line
    - cell_lines: list of cell line names to process (default: all keys in data_dict)
    - meta_regex: str, regex to identify meta columns to drop from X
    - **kwargs: Additional plotting parameters (n_rows, n_cols, figsize_per_subplot, etc.)

    Returns:
    - loo_results: dict of {cell_line: {'r2':..., 'mse':..., 'y_true':..., 'y_pred':...}}
    """
    if cell_lines is None:
        cell_lines = list(data_by_cell_line.keys())

    # Set up plotting defaults
    defaults = {
        "n_rows": 2,
        "n_cols": 3,
        "figsize_per_subplot": (4, 3)
    }
    config = {**defaults, **kwargs}
    n_rows = config['n_rows']
    n_cols = config['n_cols']
    figsize_per_subplot = config['figsize_per_subplot']

    loo_results = {}

    # Create figure for subplots if plotting is enabled
    if plot_results:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize_per_subplot[0] * n_cols, figsize_per_subplot[1] * n_rows))
        axes = axes.flatten()

    for i, cell_line in enumerate(cell_lines):
        df = data_by_cell_line[cell_line]
        # Drop meta columns from X
        meta_cols = df.filter(regex=meta_regex, axis=1)
        X = df.drop(columns=meta_cols.columns)
        if ycol not in df.columns:
            print(f"Target column '{ycol}' not found for {cell_line}, skipping.")
            continue
        if X.empty:
            print(f"Feature matrix X is empty for {cell_line}, skipping.")
            continue

        y = df[ycol]
        X = X.astype(float).fillna(0)
        y = y.astype(float).fillna(0)

        n_features = X.shape[1]

        loo = LeaveOneOut()
        y_true = []
        y_pred = []

        for train_index, test_index in loo.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # Clone the model for each split to avoid data leakage/state carryover
            from sklearn.base import clone
            model_instance = clone(model)
            model_instance.fit(X_train, y_train)
            pred = model_instance.predict(X_test)
            y_true.append(y_test.values[0])
            y_pred.append(pred[0])

        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        # Calculate Pearson correlation
        if len(y_true) > 1:
            pearson_corr = np.corrcoef(y_true, y_pred)[0, 1]
        else:
            pearson_corr = np.nan
        loo_results[cell_line] = {'r2': r2, 'mse': mse, 'pearson': pearson_corr, 'y_true': y_true, 'y_pred': y_pred}

        if print_stats:
            print(f"Cell line: {cell_line} | LOO R2: {r2:.3f} | LOO MSE: {mse:.3f} | Pearson: {pearson_corr:.3f} | Num features: {n_features}")

        # Plot in subplot if plotting is enabled
        if plot_results and i < len(axes):
            axes[i].scatter(y_true, y_pred, alpha=0.7)
            axes[i].plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label='y = y_pred')
            axes[i].set_xlabel('True Values')
            axes[i].set_ylabel('Predicted Values')
            axes[i].set_title(f'LOO {type(model).__name__} for {cell_line}\nR2={r2:.2f}, MSE={mse:.2f}, Pearson={pearson_corr:.2f}')
            axes[i].legend()

    # Hide unused axes and show plot
    if plot_results:
        for j in range(len(cell_lines), n_rows * n_cols):
            fig.delaxes(axes[j])
        plt.tight_layout()
        plt.show()

    return loo_results

def plot_data_value_distribution(final_filtered_non_tgt_prots, cell_lines, **kwargs):
    """
    Plots histograms of data values for each cell line after filtering.

    Args:
        final_filtered_non_tgt_prots (dict): Dictionary of DataFrames, one per cell line.
        cell_lines (list): List of cell line names.
        bins (int): Number of bins for the histogram.
        **kwargs: Additional keyword arguments for the histogram.
    """
    title_func=kwargs.pop('title_func',lambda cl: f"{cl} histogram of log2(fold change)")
    xlabel_func=kwargs.pop('xlabel_func',lambda _: "log2(fold change)")
    dict_of_prot_vals = {}

    for cell in cell_lines:
        meta_cols = final_filtered_non_tgt_prots[cell].filter(regex='meta_').columns
        dict_of_prot_vals[cell] = final_filtered_non_tgt_prots[cell].drop(columns=meta_cols).select_dtypes(include=[float, int]).values.flatten()



    plot_grid_graphs(
        dict_of_prot_vals,
        cell_lines,
        title_func=title_func,
        xlabel_func=xlabel_func,
        graph_type='hist',
        **kwargs
    )
    return dict_of_prot_vals

