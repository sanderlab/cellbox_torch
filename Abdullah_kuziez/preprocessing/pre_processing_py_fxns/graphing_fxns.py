import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import LeaveOneOut
import numpy as np
from tqdm import tqdm
from sklearn.base import clone
from tqdm import tqdm
from sklearn.base import clone

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
    meanandstd=config.pop('meanandstd',False)
    meanandstd=config.pop('meanandstd',False)

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
        if graph_type=='log_hist':
            sns.histplot(values, ax=axes[i], **config)
            axes[i].set_yscale('log')
            ylabel = 'Count (log scale)'
            
        if graph_type=='log_hist':
            sns.histplot(values, ax=axes[i], **config)
            axes[i].set_yscale('log')
            ylabel = 'Count (log scale)'
            
        axes[i].set_title(title_func(cell_line))
        axes[i].set_xlabel(xlabel_func(cell_line))
        axes[i].set_ylabel(ylabel)

        if meanandstd:
            mean = np.nanmean(values)
            std = np.nanstd(values)
            axes[i].axvline(x=mean, color='red', linestyle='--', label=f'Mean ({mean:.2f})')
            # Annotate mean and std on the plot
            textstr = f"Mean: {mean:.2f}\nStd: {std:.2f}"
            # Place the text in the upper right of the axes
            axes[i].text(
                0.98, 0.98, textstr,
                transform=axes[i].transAxes,
                fontsize=10,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
            )


        if meanandstd:
            mean = np.nanmean(values)
            std = np.nanstd(values)
            axes[i].axvline(x=mean, color='red', linestyle='--', label=f'Mean ({mean:.2f})')
            # Annotate mean and std on the plot
            textstr = f"Mean: {mean:.2f}\nStd: {std:.2f}"
            # Place the text in the upper right of the axes
            axes[i].text(
                0.98, 0.98, textstr,
                transform=axes[i].transAxes,
                fontsize=10,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
            )

    # Hide unused axes
    for j in range(len(cell_lines), n_rows * n_cols):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()
    return





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
        **kwargs
    )
    return dict_of_prot_vals

def filterAbyfilterB(filterfxnA,filteredfxnB,data_by_cell_line,cell_lines,kwargsA=None,kwargsB=None,**kwargs):
    """takes in two arbitrary filtering functions A and B, and their keywords, kwargsA and kwargsB, and then runs the filtering functions on the data_by_cell_line,
    using the same set of data for both. The functions are ran verbose and the metrics of interest are collected and graphed as a filterA vs filterB scatter plot
    The function does NOT work for filter incomplete experiments b/c it operates on experiments,"""
    #FUNCTION has been adjusted to work for filter outlier proteins
    verbose={'verbose':True}
    configA={**verbose,**kwargsA}
    configB={**verbose,**kwargsB}
    data_dicts_filter_A=filterfxnA(data_by_cell_line,cell_lines,**configA)
    data_dicts_filter_B=filteredfxnB(data_by_cell_line,cell_lines,**configB)
    

    graphin_dict_before_A=data_dicts_filter_A['graphing_dict_before']
    graphin_dict_before_B=data_dicts_filter_B['graphing_dict_before']

    #This code is inserted because the threshold is calculated internally in the filtering functions
    #it is generally in that case a list as well since each cell line has a different threshold
    thresh_A,thresh_B=kwargsA.get('threshold',None),kwargsB.get('threshold',None)
    thresh_A = [thresh_A] * len(cell_lines) if thresh_A is not None else None
    thresh_B = [thresh_B] * len(cell_lines) if thresh_B is not None else None
    if 'threshold' in data_dicts_filter_A.keys():
        thresh_A=data_dicts_filter_A['threshold']
    if 'threshold' in data_dicts_filter_B.keys():
        thresh_B=data_dicts_filter_B['threshold']


    # Arrange scatter plots for each cell line into subplots and add threshold lines
    import math

    n_cells = len(cell_lines)
    n_cols = kwargs.get('n_cols', 3)
    n_rows = kwargs.get('n_rows', math.ceil(n_cells / n_cols))
    figsize_per_subplot = kwargs.get('figsize_per_subplot', (4, 3))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize_per_subplot[0] * n_cols, figsize_per_subplot[1] * n_rows))
    axes = axes.flatten()

    for i, cell in enumerate(cell_lines):
        overlap_idx=list(set(graphin_dict_before_A[cell].index).intersection(set(graphin_dict_before_B[cell].index)))
        values_A = graphin_dict_before_A[cell].loc[overlap_idx]
        values_B = graphin_dict_before_B[cell].loc[overlap_idx]

        ax = axes[i]
        ax.scatter(values_A, values_B, alpha=.2,s=20)
        ax.set_xlabel('filterA')
        ax.set_ylabel('filterB')
        ax.set_title(f'{cell}')
        if kwargs.get('ylim',None) is not None:
            ax.set_ylim(kwargs.get('ylim'))
        if kwargs.get('xlim',None) is not None:
            ax.set_xlim(kwargs.get('xlim'))
        # Add threshold lines

        ax.axvline(x=thresh_A[i], color='red', linestyle='--', label=f'A threshold ({thresh_A[i]})')
        ax.axhline(y=thresh_B[i], color='blue', linestyle='--', label=f'B threshold ({thresh_B[i]})')
        if kwargs.get('xlabel',None) is not None:
            ax.set_xlabel(kwargs.get('xlabel'))
        if kwargs.get('ylabel',None) is not None:
            ax.set_ylabel(kwargs.get('ylabel'))

    # Hide unused axes
    for j in range(len(cell_lines), n_rows * n_cols):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
    return
    


def plot_protein_correlation_heatmaps(data_by_cell_line, cell_lines=None, figsize_per_subplot=(6, 5), cmap='coolwarm', vmin=-1, vmax=1, **kwargs):
    """
    Plots a heatmap of protein-protein correlations for each cell line.

    Parameters:
    - data_by_cell_line: dict
        Dictionary where keys are cell line names and values are DataFrames (rows: experiments, columns: proteins).
    - cell_lines: list or None
        List of cell line names to plot. If None, uses all keys from data_by_cell_line.
    - figsize_per_subplot: tuple
        Size of each subplot (width, height).
    - cmap: str
        Colormap for the heatmap.
    - vmin, vmax: float
        Min and max values for the colormap.
    - **kwargs: passed to seaborn.heatmap
    """
    import math
    import matplotlib.pyplot as plt
    import seaborn as sns

    if cell_lines is None:
        cell_lines = list(data_by_cell_line.keys())
    n_cells = len(cell_lines)
    n_cols = kwargs.pop('n_cols', 3)
    n_rows = kwargs.pop('n_rows', math.ceil(n_cells / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize_per_subplot[0] * n_cols, figsize_per_subplot[1] * n_rows))
    axes = axes.flatten() if n_cells > 1 else [axes]

    for i, cell in enumerate(cell_lines):
        meta_cols = data_by_cell_line[cell].filter(regex='meta_').columns
        df = data_by_cell_line[cell].drop(columns=meta_cols)
        # Only keep numeric columns (proteins)
        df_numeric = df.select_dtypes(include=[float, int])
        corr = df_numeric.corr()
        ax = axes[i]
        sns.heatmap(corr, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, cbar=True, 
                    xticklabels=False, yticklabels=False, **kwargs)
        ax.set_title(f"{cell} protein correlation")
    # Hide unused axes
    for j in range(len(cell_lines), n_rows * n_cols):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()
    return


#plotting fxns
import pandas as pd

def step_dicts_to_summary_df(*step_dicts):
    """
    Convert a sequence of step dictionaries into a DataFrame with cell lines as columns
    and filtering steps as rows (index).
    
    Parameters:
    -----------
    *step_dicts : variable number of step dictionaries
        Each step_dict should have format: {'step_name': str, 'protein_counts': {cell_line: count}}
    
    Returns:
    --------
    df : pd.DataFrame
        DataFrame with index as step names and columns as cell lines, values are protein counts.
    """
    # Filter out None step_dicts
    filtered_step_dicts = [step_dict for step_dict in step_dicts if step_dict is not None]
    if not filtered_step_dicts:
        raise ValueError("No valid step dictionaries provided (all were None).")
    
    # Collect all unique cell lines
    all_cell_lines = set()
    for step_dict in filtered_step_dicts:
        all_cell_lines.update(step_dict['protein_counts'].keys())
    all_cell_lines = sorted(list(all_cell_lines))
    
    # Build a list of dicts for DataFrame construction
    data = []
    step_names = []
    for step_dict in filtered_step_dicts:
        row = {cell_line: step_dict['protein_counts'].get(cell_line, 0) for cell_line in all_cell_lines}
        data.append(row)
        step_names.append(step_dict['step_name'])
    
    df = pd.DataFrame(data, index=step_names, columns=all_cell_lines)
    return df

def plot_filtering_progress_from_df(df, **kwargs):
    """
    Plot protein count progression across filtering steps for each cell line using a DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with index as step names and columns as cell lines, values are protein counts.
    **kwargs : optional plotting parameters
        figsize : tuple, default (12, 6)
        title : str, default 'Protein Count Progression Across Filtering Steps'
        xlabel : str, default 'Filtering Steps'
        ylabel : str, default 'Number of Proteins'
        colors : list, colors for each cell line
        markers : list, markers for each cell line
        linewidth : float, default 2
        markersize : float, default 8
        grid : bool, default True
        legend_loc : str, default 'upper right'
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    # Set default plotting parameters
    defaults = {
        'figsize': (12, 6),
        'title': 'Protein Count Progression Across Filtering Steps',
        'xlabel': 'Filtering Steps',
        'ylabel': 'Number of Proteins',
        'colors': None,
        'markers': None,
        'linewidth': 2,
        'markersize': 8,
        'grid': True,
        'legend_loc': 'upper right'
    }
    config = {**defaults, **kwargs}
    
    all_cell_lines = list(df.columns)
    step_names = list(df.index)
    
    # Set up colors and markers
    import matplotlib.pyplot as plt
    if config['colors'] is None:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        config['colors'] = (colors * ((len(all_cell_lines) // len(colors)) + 1))[:len(all_cell_lines)]
    if config['markers'] is None:
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x']
        config['markers'] = (markers * ((len(all_cell_lines) // len(markers)) + 1))[:len(all_cell_lines)]
    
    fig, ax = plt.subplots(figsize=config['figsize'])
    
    for i, cell_line in enumerate(all_cell_lines):
        ax.plot(step_names, df[cell_line].values, 
                color=config['colors'][i], 
                marker=config['markers'][i],
                linewidth=config['linewidth'],
                markersize=config['markersize'],
                label=cell_line)
    
    ax.set_title(config['title'], fontsize=14, fontweight='bold')
    ax.set_xlabel(config['xlabel'], fontsize=12)
    ax.set_ylabel(config['ylabel'], fontsize=12)
    
    if config['grid']:
        ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45, ha='right')
    ax.legend(loc=config['legend_loc'], bbox_to_anchor=(1.05, 1), borderaxespad=0)
    plt.tight_layout()
    return fig, ax





    # INSERT_YOUR_CODE

from tabulate import tabulate
def print_summary_df(summary_df, cell_lines=None):
    print(tabulate(summary_df, headers='keys', tablefmt='grid'))
    return 









def filterAbyfilterB(filterfxnA,filteredfxnB,data_by_cell_line,cell_lines,kwargsA=None,kwargsB=None,**kwargs):
    """takes in two arbitrary filtering functions A and B, and their keywords, kwargsA and kwargsB, and then runs the filtering functions on the data_by_cell_line,
    using the same set of data for both. The functions are ran verbose and the metrics of interest are collected and graphed as a filterA vs filterB scatter plot
    The function does NOT work for filter incomplete experiments b/c it operates on experiments,"""
    #FUNCTION has been adjusted to work for filter outlier proteins
    verbose={'verbose':True}
    configA={**verbose,**kwargsA}
    configB={**verbose,**kwargsB}
    data_dicts_filter_A=filterfxnA(data_by_cell_line,cell_lines,**configA)
    data_dicts_filter_B=filteredfxnB(data_by_cell_line,cell_lines,**configB)
    

    graphin_dict_before_A=data_dicts_filter_A['graphing_dict_before']
    graphin_dict_before_B=data_dicts_filter_B['graphing_dict_before']

    #This code is inserted because the threshold is calculated internally in the filtering functions
    #it is generally in that case a list as well since each cell line has a different threshold
    thresh_A,thresh_B=kwargsA.get('threshold',None),kwargsB.get('threshold',None)
    thresh_A = [thresh_A] * len(cell_lines) if thresh_A is not None else None
    thresh_B = [thresh_B] * len(cell_lines) if thresh_B is not None else None
    if 'threshold' in data_dicts_filter_A.keys():
        thresh_A=data_dicts_filter_A['threshold']
    if 'threshold' in data_dicts_filter_B.keys():
        thresh_B=data_dicts_filter_B['threshold']


    # Arrange scatter plots for each cell line into subplots and add threshold lines
    import math

    n_cells = len(cell_lines)
    n_cols = kwargs.get('n_cols', 3)
    n_rows = kwargs.get('n_rows', math.ceil(n_cells / n_cols))
    figsize_per_subplot = kwargs.get('figsize_per_subplot', (4, 3))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize_per_subplot[0] * n_cols, figsize_per_subplot[1] * n_rows))
    axes = axes.flatten()

    for i, cell in enumerate(cell_lines):
        overlap_idx=list(set(graphin_dict_before_A[cell].index).intersection(set(graphin_dict_before_B[cell].index)))
        values_A = graphin_dict_before_A[cell].loc[overlap_idx]
        values_B = graphin_dict_before_B[cell].loc[overlap_idx]

        ax = axes[i]
        ax.scatter(values_A, values_B, alpha=.2,s=20)
        ax.set_xlabel('filterA')
        ax.set_ylabel('filterB')
        ax.set_title(f'{cell}')
        if kwargs.get('ylim',None) is not None:
            ax.set_ylim(kwargs.get('ylim'))
        if kwargs.get('xlim',None) is not None:
            ax.set_xlim(kwargs.get('xlim'))
        # Add threshold lines

        ax.axvline(x=thresh_A[i], color='red', linestyle='--', label=f'A threshold ({thresh_A[i]})')
        ax.axhline(y=thresh_B[i], color='blue', linestyle='--', label=f'B threshold ({thresh_B[i]})')
        if kwargs.get('xlabel',None) is not None:
            ax.set_xlabel(kwargs.get('xlabel'))
        if kwargs.get('ylabel',None) is not None:
            ax.set_ylabel(kwargs.get('ylabel'))

    # Hide unused axes
    for j in range(len(cell_lines), n_rows * n_cols):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
    return
    


def plot_protein_correlation_heatmaps(data_by_cell_line, cell_lines=None, figsize_per_subplot=(6, 5), cmap='coolwarm', vmin=-1, vmax=1, **kwargs):
    """
    Plots a heatmap of protein-protein correlations for each cell line.

    Parameters:
    - data_by_cell_line: dict
        Dictionary where keys are cell line names and values are DataFrames (rows: experiments, columns: proteins).
    - cell_lines: list or None
        List of cell line names to plot. If None, uses all keys from data_by_cell_line.
    - figsize_per_subplot: tuple
        Size of each subplot (width, height).
    - cmap: str
        Colormap for the heatmap.
    - vmin, vmax: float
        Min and max values for the colormap.
    - **kwargs: passed to seaborn.heatmap
    """
    import math
    import matplotlib.pyplot as plt
    import seaborn as sns

    if cell_lines is None:
        cell_lines = list(data_by_cell_line.keys())
    n_cells = len(cell_lines)
    n_cols = kwargs.pop('n_cols', 3)
    n_rows = kwargs.pop('n_rows', math.ceil(n_cells / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize_per_subplot[0] * n_cols, figsize_per_subplot[1] * n_rows))
    axes = axes.flatten() if n_cells > 1 else [axes]

    for i, cell in enumerate(cell_lines):
        meta_cols = data_by_cell_line[cell].filter(regex='meta_').columns
        df = data_by_cell_line[cell].drop(columns=meta_cols)
        # Only keep numeric columns (proteins)
        df_numeric = df.select_dtypes(include=[float, int])
        corr = df_numeric.corr()
        ax = axes[i]
        sns.heatmap(corr, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, cbar=True, 
                    xticklabels=False, yticklabels=False, **kwargs)
        ax.set_title(f"{cell} protein correlation")
    # Hide unused axes
    for j in range(len(cell_lines), n_rows * n_cols):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()
    return


#plotting fxns
import pandas as pd

def step_dicts_to_summary_df(*step_dicts):
    """
    Convert a sequence of step dictionaries into a DataFrame with cell lines as columns
    and filtering steps as rows (index).
    
    Parameters:
    -----------
    *step_dicts : variable number of step dictionaries
        Each step_dict should have format: {'step_name': str, 'protein_counts': {cell_line: count}}
    
    Returns:
    --------
    df : pd.DataFrame
        DataFrame with index as step names and columns as cell lines, values are protein counts.
    """
    # Filter out None step_dicts
    filtered_step_dicts = [step_dict for step_dict in step_dicts if step_dict is not None]
    if not filtered_step_dicts:
        raise ValueError("No valid step dictionaries provided (all were None).")
    
    # Collect all unique cell lines
    all_cell_lines = set()
    for step_dict in filtered_step_dicts:
        all_cell_lines.update(step_dict['protein_counts'].keys())
    all_cell_lines = sorted(list(all_cell_lines))
    
    # Build a list of dicts for DataFrame construction
    data = []
    step_names = []
    for step_dict in filtered_step_dicts:
        row = {cell_line: step_dict['protein_counts'].get(cell_line, 0) for cell_line in all_cell_lines}
        data.append(row)
        step_names.append(step_dict['step_name'])
    
    df = pd.DataFrame(data, index=step_names, columns=all_cell_lines)
    return df

def plot_filtering_progress_from_df(df, **kwargs):
    """
    Plot protein count progression across filtering steps for each cell line using a DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with index as step names and columns as cell lines, values are protein counts.
    **kwargs : optional plotting parameters
        figsize : tuple, default (12, 6)
        title : str, default 'Protein Count Progression Across Filtering Steps'
        xlabel : str, default 'Filtering Steps'
        ylabel : str, default 'Number of Proteins'
        colors : list, colors for each cell line
        markers : list, markers for each cell line
        linewidth : float, default 2
        markersize : float, default 8
        grid : bool, default True
        legend_loc : str, default 'upper right'
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    # Set default plotting parameters
    defaults = {
        'figsize': (12, 6),
        'title': 'Protein Count Progression Across Filtering Steps',
        'xlabel': 'Filtering Steps',
        'ylabel': 'Number of Proteins',
        'colors': None,
        'markers': None,
        'linewidth': 2,
        'markersize': 8,
        'grid': True,
        'legend_loc': 'upper right'
    }
    config = {**defaults, **kwargs}
    
    all_cell_lines = list(df.columns)
    step_names = list(df.index)
    
    # Set up colors and markers
    import matplotlib.pyplot as plt
    if config['colors'] is None:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        config['colors'] = (colors * ((len(all_cell_lines) // len(colors)) + 1))[:len(all_cell_lines)]
    if config['markers'] is None:
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x']
        config['markers'] = (markers * ((len(all_cell_lines) // len(markers)) + 1))[:len(all_cell_lines)]
    
    fig, ax = plt.subplots(figsize=config['figsize'])
    
    for i, cell_line in enumerate(all_cell_lines):
        ax.plot(step_names, df[cell_line].values, 
                color=config['colors'][i], 
                marker=config['markers'][i],
                linewidth=config['linewidth'],
                markersize=config['markersize'],
                label=cell_line)
    
    ax.set_title(config['title'], fontsize=14, fontweight='bold')
    ax.set_xlabel(config['xlabel'], fontsize=12)
    ax.set_ylabel(config['ylabel'], fontsize=12)
    
    if config['grid']:
        ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45, ha='right')
    ax.legend(loc=config['legend_loc'], bbox_to_anchor=(1.05, 1), borderaxespad=0)
    plt.tight_layout()
    return fig, ax





    # INSERT_YOUR_CODE

from tabulate import tabulate
def print_summary_df(summary_df, cell_lines=None):
    print(tabulate(summary_df, headers='keys', tablefmt='grid'))
    return 









