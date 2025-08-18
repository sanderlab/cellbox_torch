import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
from .graphing_fxns import step_dicts_to_summary_df
from .graphing_fxns import plot_grid_graphs
from .graphing_fxns import print_summary_df
from .graphing_fxns import plot_filtering_progress_from_df
import copy
from sklearn.model_selection import LeaveOneOut
import numpy as np
from tqdm import tqdm
from sklearn.base import clone
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import ElasticNetCV
import random
import string
from pathlib import Path
import hashlib

#/Basic incompleteness filters

def load_data(intermediate_dir, timepoint): 
    """returns a dict that has all the variables inside it
    loads the data of the files after restructuring the data from the intermediate files folder"""
    time_point_dict={}
    with open(intermediate_dir / f"data_by_cell_line_raw_{timepoint}.pkl", "rb") as f:
        time_point_dict['data_by_cell_line_raw'] = pickle.load(f)
    with open(intermediate_dir / f"control_data_by_cell_line_{timepoint}.pkl", "rb") as f:
        time_point_dict['control_data_by_cell_line'] = pickle.load(f)
    with open(intermediate_dir / f"control_data_by_cell_line_coeffvar_{timepoint}.pkl", "rb") as f:
        time_point_dict['control_data_by_cell_line_coeffvar'] = pickle.load(f)
    with open(intermediate_dir / f"targeted_prots_raw_{timepoint}.pkl", "rb") as f:
        time_point_dict['targeted_prots_raw'] = pickle.load(f)
    with open(intermediate_dir / f"non_targeted_prots_raw_{timepoint}.pkl", "rb") as f:
        time_point_dict['non_targeted_prots_raw'] = pickle.load(f)
    with open(intermediate_dir / f"cell_lines_{timepoint}.pkl", "rb") as f:
        time_point_dict['cell_lines'] = pickle.load(f)
    with open(intermediate_dir / f"drug_pert_id_targets_dict_{timepoint}.pkl", "rb") as f:
        time_point_dict['drug_pert_id_targets_dict'] = pickle.load(f)
    with open(intermediate_dir / f"symbol_to_uniprot_{timepoint}.json", "r") as f:
        time_point_dict['symbol_to_uniprot'] = json.load(f)

    return time_point_dict

class filtering_pipeline:
    def __init__(self,data_by_cell_line,cell_lines,control_data_by_cell_line,timepoint,coeffvar_by_cell_line,**kwargs):
        self.data_by_cell_line=data_by_cell_line
        self.cell_lines=cell_lines
        self.control_data_by_cell_line=control_data_by_cell_line
        self.timepoint=timepoint
        self.config={
            'print_flag':kwargs.pop('print_flag',False),
            'graph_flag':kwargs.pop('graph_flag',False),
            'graph_type':kwargs.pop('graph_type','hist'),
            'filter_flag':kwargs.pop('filter_flag',True),
            'verbose':kwargs.pop('verbose',True),
        }
        self.coeffvar_by_cell_line=coeffvar_by_cell_line
        self.summary_df=None
        self.pipeline_hash=None
        self.pipeline_used=None
        self.log_2_transformed_data=None

    def set_config(self,**kwargs):
        self.config.update(kwargs)


    #intakes a list of filters, then runs them in order and passes the output of each filter to the next 
    def run_pipeline(self, filter_list, save_dir=None, tgt=None):
        """
        Run the filtering pipeline. If save_dir and tgt are provided, check for existing
        saved pipeline with the same configuration and load it if found.
        
        Args:
            filter_list: List of filters to apply
            save_dir: Directory to check for/save existing pipeline results
            tgt: Target status ('tgt' or 'nontgt') for file naming
            
        Returns:
            outputs_from_steps: Dictionary containing results from each filtering step
        """
        
        self.pipeline_used = filter_list
        self.hash_value = self._generate_pipeline_hash()
        
        # Check for existing saved pipeline if save_dir and tgt are provided
        if save_dir is not None and tgt is not None:
            # Generate hash for this configuration
            file_ID = self._generate_pipeline_hash()
            
            # Convert to Path object for consistent handling
            save_dir = Path(save_dir)
            filename = f"{self.timepoint}_{tgt}_{file_ID}_pipeline.pkl"
            file_path = save_dir / filename
            
            # Check if file already exists
            if file_path.exists():
                print(f"🔍 Found existing pipeline with identical configuration!")
                print(f"   File: {file_path}")
                print(f"   Configuration hash: {file_ID}")
                print(f"   Loading existing results instead of rerunning...")
                
                try:
                    # Load the existing pipeline
                    loaded_pipeline = self.load_pipeline(file_path)
                    
                    # Copy the relevant attributes from the loaded pipeline
                    self.final_filtered_data = loaded_pipeline.final_filtered_data
                    self.outputs_from_steps = loaded_pipeline.outputs_from_steps
                    self.pipeline_used = loaded_pipeline.pipeline_used
                    
                    print(f"✅ Successfully loaded existing pipeline results!")
                    return self.outputs_from_steps
                    
                except Exception as e:
                    print(f"⚠️ Error loading existing pipeline: {e}")
                    print("   Proceeding to run pipeline from scratch...")
        
        outputs_from_steps = {}
        filtered_data = self.data_by_cell_line

        raw_result = {
            'filtered_data': self.data_by_cell_line,
            'graphing_dict_before': {},
            'graphing_dict_after': {},
            'step_dict': {'step_name': 'original_data', 'protein_counts': {cell: len(self.data_by_cell_line[cell].columns) for cell in self.cell_lines}}
        }
        outputs_from_steps['step_0_original_data'] = raw_result

        for i, filter in enumerate(filter_list):
            print(f'applying filter {i+1} of {len(filter_list)}: {filter}')
            
            if filter[0] == 'filter_proteins_with_control':
                result = filter_proteins_with_control(filtered_data, self.cell_lines, self.control_data_by_cell_line, **self.config)
                filtered_data, graphing_dict_before, graphing_dict_after, step_dict = result['filtered_data'], result['graphing_dict_before'], result['graphing_dict_after'], result['step_dict']
 
            elif filter[0] == 'log2_transform_by_control':
                result = log2_transform_by_control(filtered_data, self.cell_lines, self.control_data_by_cell_line, **self.config)
                filtered_data, graphing_dict_before, graphing_dict_after, step_dict = result['filtered_data'], result['graphing_dict_before'], result['graphing_dict_after'], result['step_dict']
                self.log_2_transformed_data=filtered_data
            elif filter[0] == 'filter_incomplete_proteins':
                result = filter_incomplete_proteins(filtered_data, self.cell_lines, completeness_threshold_prot=float(filter[1]), **self.config)
                filtered_data, graphing_dict_before, graphing_dict_after, step_dict = result['filtered_data'], result['graphing_dict_before'], result['graphing_dict_after'], result['step_dict']
           
            elif filter[0] == 'filter_incomplete_experiments':
                result = filter_incomplete_experiments(filtered_data, self.cell_lines, completeness_threshold_experiment=float(filter[1]), **self.config)
                filtered_data, graphing_dict_before, graphing_dict_after, step_dict = result['filtered_data'], result['graphing_dict_before'], result['graphing_dict_after'], result['step_dict']
           
            elif filter[0] == 'filter_keep_low_cv':
                result = filter_keep_low_cv(filtered_data, self.cell_lines, max_cv=float(filter[1]), coeffvar_by_cell_line=self.coeffvar_by_cell_line, **self.config)
                filtered_data, graphing_dict_before, graphing_dict_after, step_dict = result['filtered_data'], result['graphing_dict_before'], result['graphing_dict_after'], result['step_dict']
           
            elif filter[0] == 'remove_outlier_proteins':
                result = remove_outlier_proteins(filtered_data, self.cell_lines, outlier_factor=float(filter[1]), **self.config)
                filtered_data, graphing_dict_before, graphing_dict_after, step_dict = result['filtered_data'], result['graphing_dict_before'], result['graphing_dict_after'], result['step_dict']
           
            elif filter[0] == 'filter_by_mutual_information':
                result = filter_by_mutual_information(filtered_data, self.cell_lines, mi_thresh=float(filter[1]), y_col=filter[2], **self.config)
                filtered_data, graphing_dict_before, graphing_dict_after, step_dict = result['filtered_data'], result['graphing_dict_before'], result['graphing_dict_after'], result['step_dict']
           
            elif filter[0] == 'iterative_signal_filtering':
                result = iterative_signal_filtering(filtered_data, self.cell_lines, std_threshold=float(filter[1]), **self.config)
                filtered_data, graphing_dict_before, graphing_dict_after, step_dict = result['filtered_data'], result['graphing_dict_before'], result['graphing_dict_after'], result['step_dict']
           
            elif filter[0] == 'spearman_corr_filtering':
                result = spearman_corr_filtering(filtered_data, self.cell_lines, threshold=float(filter[1]), ycol=filter[2], **self.config)
                filtered_data, graphing_dict_before, graphing_dict_after, step_dict = result['filtered_data'], result['graphing_dict_before'], result['graphing_dict_after'], result['step_dict']
           
            elif filter[0] == 'pearson_corr_filtering':
                result = pearson_corr_filtering(filtered_data, self.cell_lines, threshold=float(filter[1]), ycol=filter[2], **self.config)
                filtered_data, graphing_dict_before, graphing_dict_after, step_dict = result['filtered_data'], result['graphing_dict_before'], result['graphing_dict_after'], result['step_dict']
           
            elif filter[0] == 'fill_na_with_mean':
                result = fill_na_with_mean(filtered_data, self.cell_lines, **self.config)
                filtered_data, graphing_dict_before, graphing_dict_after, step_dict = result['filtered_data'], result['graphing_dict_before'], result['graphing_dict_after'], result['step_dict']
            elif filter[0]== 'inject_noise':
                result = self.inject_noise(filtered_data, self.cell_lines, mean=float(filter[1]),sigma=float(filter[2]), **self.config)
                filtered_data, graphing_dict_before, graphing_dict_after, step_dict = result['filtered_data'], result['graphing_dict_before'], result['graphing_dict_after'], result['step_dict']
            else:
                print(f'filter {filter} not found')
           
            outputs_from_steps[f'step_{i+1}_{filter[0]}_{filter[1]}'] = (result)
            
        self.final_filtered_data = filtered_data
        self.outputs_from_steps = outputs_from_steps

        return outputs_from_steps
    
    def make_summary_df(self):
        # Check if required attributes exist
        if not hasattr(self, 'outputs_from_steps') or self.outputs_from_steps is None:
            raise ValueError("No pipeline outputs found. Run pipeline first using run_pipeline().")
        if not hasattr(self, 'cell_lines') or self.cell_lines is None:
            raise ValueError("Cell lines not defined.")
        
        self.pipeline_hash = self._generate_pipeline_hash()
        summary_df = pd.DataFrame(columns=self.cell_lines)
        
        for step_key in self.outputs_from_steps.keys():
            data_dict = self.outputs_from_steps[step_key]['filtered_data']

            if step_key == 'step_0_original_data':
                condition_name = 'raw_data'
            else:
                condition_name=step_key
            
            # Count proteins for each cell line
            for cell in self.cell_lines:
                if cell in data_dict:
                    # Count only non-meta columns to be consistent with other functions
                    non_meta_cols = [col for col in data_dict[cell].columns if not col.startswith('meta_')]
                    summary_df.loc[condition_name, cell] = len(non_meta_cols)
                else:
                    summary_df.loc[condition_name, cell] = 0
        
        self.summary_df = summary_df
        return summary_df
 
    def plot_filtering_progress(self):
        fig, ax=plot_filtering_progress_from_df(self.summary_df,title=f'{self.timepoint} protein count progression across filtering steps')
        ax.text(
            1.0, 1.05, f"Hash: {self.pipeline_hash}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )
        return fig, ax

    def summary_of_pipeline(self):
        self.pipeline_hash=self._generate_pipeline_hash()
        print(f'////////////Hash:   {self.pipeline_hash}   ///////////////////////////')
        self.make_summary_df()
        print_summary_df(self.summary_df)
        adjusted_title={
            'title':f'{self.timepoint} protein count progression across filtering steps',
        }
        fig, ax = plot_filtering_progress_from_df(self.summary_df, **adjusted_title)

        return self.summary_df

    def _generate_pipeline_hash(self):

        # Create a string representation of the key pipeline parameters
        hash_components = []
        
        # Add timepoint
        hash_components.append(f"timepoint:{self.timepoint}")
        
        # Add config parameters
        for key, value in sorted(self.config.items()):
            hash_components.append(f"{key}:{value}")
        
        # Add pipeline_used if it exists
        if hasattr(self, 'pipeline_used'):
            for filter in self.pipeline_used:
                hash_components.append(f"{filter[0]}:{filter[1]}")

        # Create the full hash string
        hash_string = '|'.join(hash_components)
        
        # Generate MD5 hash and take first 6 characters
        hash_obj = hashlib.md5(hash_string.encode('utf-8'))
        hash_hex = hash_obj.hexdigest()
        
        return hash_hex[:6]

    def save_pipeline(self, save_dir, tgt=None, file_ID=None):
     
        if tgt is None:
            raise ValueError("Please select tgt status")
        
        # Convert to Path object for consistent handling
        save_dir = Path(save_dir)
        
        # Create directory if it doesn't exist
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate hash-based file_ID if not provided
        if file_ID is None:
            file_ID = self._generate_pipeline_hash()
        
        # Create the full file path
        filename = f"{self.timepoint}_{tgt}_{file_ID}_pipeline.pkl"
        file_path = save_dir / filename
        
        # # Check if file already exists
        # if file_path.exists():
        #     print(f"⚠️  WARNING: A pipeline with identical parameters already exists!")
        #     print(f"   File: {file_path}")
        #     print(f"   This suggests you're trying to save the same pipeline configuration.")
            
        #     # Ask user what they want to do
        #     response = input("Do you want to overwrite the existing file? (y/N): ").lower().strip()
        #     if response not in ['y', 'yes']:
        #         print("Save operation cancelled.")
        #         return file_ID
        
        # Save the pipeline
        try:
            with open(file_path, "wb") as f:
                pickle.dump(self, f)
            if file_path.exists():
                print(f'✅ Filtering pipeline saved to {file_path}')
            else:
                print(f'✅ Filtering pipeline overwritten at {file_path}')
            return file_ID
        except Exception as e:
            print(f"❌ Error saving pipeline: {e}")
            raise
    
    def run_loo_regression(self,ycol,model,**kwargs):
        """intakes a ycol, model, and then runs a loo regression for the model and the ycol
        optionally intakes a data_by_cell_line, cell_lines, print_stats, plot_results, show_progress, and hash_value"""
        loo_results = loo_regression_per_cell_line(
        data_by_cell_line=kwargs.get('data_by_cell_line',self.final_filtered_data),
        cell_lines=kwargs.get('cell_lines',self.cell_lines),
        ycol=ycol,  # Use the passed parameter directly
        model=model,  # Use the passed parameter directly
        print_stats=kwargs.get('print_stats',True),
        plot_results=kwargs.get('plot_results',True),
        show_progress=kwargs.get('show_progress',True),
        hash_value=kwargs.get('hash_value',self.pipeline_hash)
        )
        return loo_results
    
    
    def inject_noise(self,mean_mult,sigma_mult,**kwargs):
        """intakes data and then injects noise per column with a mean of column_mean*mean_mult and a std of column_std*sigma_mult
        optionally intakes a noise function in kwargs that is used"""
        data_by_cell_line=self.final_filtered_data
        cell_lines=self.cell_lines
        step_dict={'protein_counts':{},'mean_mult':{},'sigma_mult':{},'noise_fxn':{}}
        noisy_data, graphing_dict_before, graphing_dict_after = {}, {}, {}
        print_flag = kwargs.pop('print_flag', False)
        verbose=kwargs.pop('verbose',False)
        graph_flag=kwargs.pop('graph_flag',False)
        noise_fxn=kwargs.pop('noise_function',np.random.normal)
        print('injecting noise')
        
        for cell in cell_lines:
            noisy_data[cell]=data_by_cell_line[cell].copy()
            #inject noise per column
            for col in noisy_data[cell].select_dtypes(include=['number']).columns:
                mean=mean_mult*noisy_data[cell][col].mean()
                std=sigma_mult*noisy_data[cell][col].std()
                noise=noise_fxn(loc=mean,scale=std,size=noisy_data[cell][col].shape)
                noisy_data[cell][col]=noisy_data[cell][col]+noise


            #count the number of proteins
            # Filter out columns containing '_meta' before counting
            protein_cols = [col for col in noisy_data[cell].columns if '_meta' not in col]
            step_dict['protein_counts'][cell] = len(protein_cols)
            step_dict['mean_mult'][cell]=mean_mult
            step_dict['sigma_mult'][cell]=sigma_mult
            step_dict['noise_fxn'][cell]=noise_fxn

            if verbose or graph_flag:
                graphing_dict_before[cell]=data_by_cell_line[cell]
                graphing_dict_after[cell]=noisy_data[cell]

        if verbose:
            return {
                'filtered_data': noisy_data,
                'graphing_dict_before': graphing_dict_before,
                'graphing_dict_after': graphing_dict_after,
                'step_dict': step_dict
            }
        self.final_filtered_data=noisy_data #updating the intenal data to the noisy data
        original_data=data_by_cell_line
        return original_data, noisy_data, step_dict
    
    def noise_stability_test(self,mean_array,sigma_array,**kwargs):
        """intakes a mean_mult and an array of stds and then runs a loo regression for the mean_mult and each std,
        after regression generates a plot of the r^2 values for std per cell line"""
        import matplotlib.pyplot as plt
        import seaborn as sns

        print('running noise stability test')
        model=kwargs.pop('model',ElasticNetCV(
            alphas=[0.001, 0.01, 0.1, 1.0, 10.0],           # Alpha values to scan
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],            # L1 ratio values to scan
            cv=5,                                            # 5-fold CV for parameter selection
            random_state=42,
            max_iter=5000
        ))


        ycol=kwargs.pop('ycol','meta_Inhi_5')
        plot_flag=kwargs.pop('plot_flag',True)
        figsize=kwargs.pop('figsize',(12, 6))

        original_data = copy.deepcopy(self.final_filtered_data)
        r2_values={cell:[] for cell in self.cell_lines}
        x_labels = []  # Store labels for x-axis
        combo_mean_sigma = list(zip(mean_array, sigma_array))
        
        for i, (mean_mult, sigma) in enumerate(combo_mean_sigma):
            print(f'running noise stability test with sigma_mult={sigma}, mean_mult={mean_mult} ({i+1}/{len(combo_mean_sigma)})')
            x_labels.append(f'μ={mean_mult:.2f},σ={sigma:.2f}')

            og_data,noisy_data,step_dict=self.inject_noise(mean_mult=mean_mult, sigma_mult=sigma, **kwargs)
            
            # Run regression with no plotting
            no_plot = {'plot_results': False}
            results = self.run_loo_regression(ycol=ycol, model=model, **no_plot)
            for cell in self.cell_lines:
                r2_values[cell].append(results[cell]['r2'])
        
            self.final_filtered_data = original_data

        if plot_flag:

            plt.figure(figsize=figsize)
            markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x']
    
            
            for i, cell in enumerate(self.cell_lines):
                plt.plot(range(len(x_labels)), r2_values[cell], 
                        marker=markers[i], linewidth=2, markersize=6,
                        label=cell)
            
            plt.xlabel('Noise Parameters (Mean Mult, Sigma Mult)', fontsize=12)
            plt.ylabel('R² Score', fontsize=12)
            plt.title('Noise Stability Test: R² Performance vs Noise Levels', fontsize=14)
            plt.xticks(range(len(x_labels)), x_labels, rotation=45, ha='right')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        
        # Return structured results
        results_dict = {
            'r2_values': r2_values,
            'mean_array': mean_array,
            'sigma_array': sigma_array,
            'x_labels': x_labels,
            'cell_lines': self.cell_lines,
            'ycol': ycol
        }
        return results_dict

    # def counting_overlaps(fp_1: "filtering_pipeline",fp_2: "filtering_pipeline"):

    #     #count and plot the number of proteins present in the pipelines variously:
    #     #1. in both pipelines
    #     #2. in pipeline 1 but not in pipeline 2
    #     #3. in pipeline 2 but not in pipeline 1
    #     #4. in neither pipeline
    #     #5. in both pipelines but different
    #     #6. in both pipelines but same
    #     pass
    
    @staticmethod
    def load_pipeline(load_path):
        load_path = Path(load_path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Pipeline file not found: {load_path}")
        
        try:
            with open(load_path, "rb") as f:
                loaded_pipeline = pickle.load(f)
            print(f'Pipeline loaded from {load_path}')
            return loaded_pipeline
        except Exception as e:
            print(f"Error loading pipeline: {e}")
            raise
    
def fill_na_with_mean(data_by_cell_line, cell_lines, **kwargs):
    """
    Fill NaN values with the mean of each protein column for each cell line.
    """
    filtered_data, graphing_dict_before, graphing_dict_after = {}, {}, {}
    print_flag = kwargs.pop('print_flag', False)
    graph_flag = kwargs.pop('graph_flag', False)
    graph_type = kwargs.pop('graph_type', 'hist')
    filter_flag = kwargs.pop('filter_flag', True)
    verbose = kwargs.pop('verbose', True)
    
    if not filter_flag:
        print('filling na with mean is disabled')
        return data_by_cell_line, None

    print('filling na with mean')
    
    # Initialize step dictionary
    step_dict = {'step_name': 'fill_na', 'protein_counts': {}}

    for cell in cell_lines:
        df = data_by_cell_line[cell].copy()
        
        # Fill NaN values with mean for non-meta columns
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mean())
        
        filtered_data[cell] = df
        
        # Count only non-meta proteins for the step dictionary
        non_meta_prots = len([col for col in df.columns if not col.startswith('meta_')])
        step_dict['protein_counts'][cell] = non_meta_prots
        
        if print_flag:
            na_before = data_by_cell_line[cell].isna().sum().sum()
            na_after = filtered_data[cell].isna().sum().sum()
            print(f"[{cell}] {na_before} NaN values -> {na_after} NaN values, {non_meta_prots} prots")

        if graph_flag or verbose:
            graphing_dict_before[cell] = data_by_cell_line[cell].isna().sum()
            graphing_dict_after[cell] = filtered_data[cell].isna().sum()

    if print_flag:
        print('\n\n')
    if graph_flag:
        # Plot before:
        plot_grid_graphs(
            graphing_dict_before,
            cell_lines,
            title_func=lambda cl: f"{cl} NaN count by protein (before)",
            xlabel_func=lambda _: "NaN count",
            graph_type=graph_type,
            **kwargs
        )
        # Plot after:
        plot_grid_graphs(
            graphing_dict_after,
            cell_lines,
            title_func=lambda cl: f"{cl} NaN count by protein (after)",
            xlabel_func=lambda _: "NaN count",
            graph_type=graph_type,
            **kwargs
        )
    
    if verbose:
        return {
            'filtered_data': filtered_data,
            'graphing_dict_before': graphing_dict_before,
            'graphing_dict_after': graphing_dict_after,
            'step_dict': step_dict
        }
    return filtered_data, step_dict

def filter_proteins_with_control(data_by_cell_line, cell_lines, control_data_by_cell_line, **kwargs):
    filtered_data,graphing_dict_before,graphing_dict_after={},{},{}
    print_flag=kwargs.pop('print_flag',False)
    graph_flag=kwargs.pop('graph_flag',False)
    graph_type=kwargs.pop('graph_type','hist')
    filter_flag=kwargs.pop('filter_flag',True)
    verbose=kwargs.pop('verbose',True)
    if not filter_flag:
        print('filtering proteins with control values is disabled')
        return data_by_cell_line, None
    print('filtering proteins with control values')
    
    # Initialize step dictionary
    step_dict = {'step_name': 'no_ctrl', 'protein_counts': {}}
    
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
        
        # Count only non-meta proteins for the step dictionary
        non_meta_prots = len([col for col in control_prots_intersect if not col.startswith('meta_')])
        step_dict['protein_counts'][cell] = non_meta_prots
        
        if print_flag:
            print(f"[{cell}] {len(data_prots)} prots -> {non_meta_prots} prots")

        if graph_flag or verbose:
            graphing_dict_before[cell] = pd.Series(
            [1 if prot in control_prots_intersect else 0 for prot in data_by_cell_line[cell].columns],
            index=data_by_cell_line[cell].columns
            )
            graphing_dict_after[cell] = pd.Series(
            [1 if prot in control_prots_intersect else 0 for prot in filtered_data[cell].columns],
            index=filtered_data[cell].columns
            )
    if print_flag:
        print('\n\n')
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

    if verbose:
        return {
            'filtered_data': filtered_data,
            'graphing_dict_before': graphing_dict_before,
            'graphing_dict_after': graphing_dict_after,
            'step_dict': step_dict
        }
    return filtered_data, step_dict

def filter_incomplete_proteins(data_by_cell_line, cell_lines, completeness_threshold_prot=0.8, **kwargs):
    filtered_data,graphing_dict_before,graphing_dict_after={},{},{}
    print_flag=kwargs.pop('print_flag',False)
    graph_flag=kwargs.pop('graph_flag',False)
    graph_type=kwargs.pop('graph_type','hist')
    filter_flag=kwargs.pop('filter_flag',True)
    verbose=kwargs.pop('verbose',True)
    if not filter_flag:
        print('filtering incomplete proteins is disabled')
        return data_by_cell_line, None

    print('filtering incomplete proteins')
    
    # Initialize step dictionary
    step_dict = {'step_name': 'incomp_prot', 'protein_counts': {}}

    for cell_line in cell_lines:
        df = data_by_cell_line[cell_line]
        before_cols = df.shape[1]
        n_trials = df.shape[0]
        min_non_na = int(completeness_threshold_prot * n_trials)

        filtered_df = df.dropna(axis=1, thresh=min_non_na)
        filtered_data[cell_line] = filtered_df
        after_cols = filtered_df.shape[1]
        
        # Count only non-meta proteins for the step dictionary
        non_meta_prots = len([col for col in filtered_df.columns if not col.startswith('meta_')])
        step_dict['protein_counts'][cell_line] = non_meta_prots
        
        if print_flag:
            print(f"[{cell_line}] {before_cols} cols -> {non_meta_prots} prots")
        if graph_flag or verbose:
            graphing_dict_before[cell_line]=df.isna().sum(axis=0)/df.shape[0]
            graphing_dict_after[cell_line]=filtered_df.isna().sum(axis=0)/filtered_df.shape[0]
    if print_flag:
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
    
    if verbose:
        return {
            'filtered_data': filtered_data,
            'graphing_dict_before': graphing_dict_before,
            'graphing_dict_after': graphing_dict_after,
            'step_dict': step_dict
        }
    return filtered_data, step_dict

def filter_incomplete_experiments(data_by_cell_line, cell_lines, completeness_threshold_experiment=0.8, **kwargs):
    filtered_data,graphing_dict_before,graphing_dict_after={},{},{}
    print_flag=kwargs.pop('print_flag',False)
    graph_flag=kwargs.pop('graph_flag',False)
    graph_type=kwargs.pop('graph_type','hist')
    filter_flag=kwargs.pop('filter_flag',True)
    verbose=kwargs.pop('verbose',True)
    if not filter_flag:
        print('filtering incomplete experiments is disabled')
        return data_by_cell_line, None

    print('filtering incomplete experiments')
    
    # Initialize step dictionary
    step_dict = {'step_name': 'incomp_exp', 'protein_counts': {}}

    for cell_line in cell_lines:
        df = data_by_cell_line[cell_line]
        before_rows = df.shape[0]
        min_non_na = int(completeness_threshold_experiment * df.shape[1])
        filtered_df = df.dropna(axis=0, thresh=min_non_na)
        filtered_data[cell_line] = filtered_df
        after_rows = filtered_df.shape[0]
        
        # Count only non-meta proteins for the step dictionary
        non_meta_prots = len([col for col in filtered_df.columns if not col.startswith('meta_')])
        step_dict['protein_counts'][cell_line] = non_meta_prots
        
        if print_flag:
            print(f"[{cell_line}] {before_rows} experiments -> {after_rows} experiments, {non_meta_prots} prots")
        if graph_flag or verbose:
            graphing_dict_before[cell_line]=df.isna().sum(axis=1)/df.shape[1]
            graphing_dict_after[cell_line]=filtered_df.isna().sum(axis=1)/filtered_df.shape[1]

    if print_flag:
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
    
    if verbose:
        return {
            'filtered_data': filtered_data,
            'graphing_dict_before': graphing_dict_before,
            'graphing_dict_after': graphing_dict_after,
            'step_dict': step_dict
        }
    return filtered_data, step_dict


#high variance filters
def filter_keep_low_cv(data_by_cell_line, cell_lines, max_cv, coeffvar_by_cell_line, **kwargs):
    graphing_dict_before,graphing_dict_after,filtered_data = {},{},{}
    print_flag=kwargs.pop('print_flag',False)
    graph_flag=kwargs.pop('graph_flag',False)
    graph_type=kwargs.pop('graph_type','hist')
    filter_flag=kwargs.pop('filter_flag',True)
    verbose=kwargs.pop('verbose',True)
    if not filter_flag:
        print('filtering low coefficient of variation proteins is disabled')
        return data_by_cell_line, None

    print('filtering low coefficient of variation proteins')
    
    # Initialize step dictionary
    step_dict = {'step_name': 'high_cv', 'protein_counts': {}}

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
        
        # Count only non-meta proteins for the step dictionary
        non_meta_prots = len([col for col in keep_cols if not col.startswith('meta_')])
        step_dict['protein_counts'][cell] = non_meta_prots
        
        if print_flag:
            print(f"[{cell}] {original_num_proteins} -> {non_meta_prots} prots")

        if graph_flag or verbose:
            # Get a pandas Series of the numeric coefficient of variation values before filtering
            col_set_1=data_by_cell_line[cell].select_dtypes(include=[float, int]).columns.intersection(set(numeric_coeff_var.columns))
            graphing_dict_before[cell] = numeric_coeff_var[col_set_1].iloc[0, :]
            intersect = list(set(keep_cols).intersection(set(coeff_var.columns)))
            # Get a pandas Series of the numeric coefficient of variation values after filtering
            graphing_dict_after[cell] = numeric_coeff_var[intersect].iloc[0, :]

    if print_flag:
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
    
    if verbose:
        return {
            'filtered_data': filtered_data,
            'graphing_dict_before': graphing_dict_before,
            'graphing_dict_after': graphing_dict_after,
            'step_dict': step_dict
        }
    return filtered_data, step_dict

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
    verbose=kwargs.pop('verbose',True)
    if not filter_flag:
        print('filtering outlier proteins is disabled')
        return data_by_cell_line, None

    print('removing outlier proteins')
    
    # Initialize step dictionary
    step_dict = {'step_name': 'outlier', 'protein_counts': {}}

    threshold_holder=[]
    for i,cell_line in enumerate(cell_lines):
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
        
        # Count only non-meta proteins for the step dictionary (consistent with other functions)
        non_meta_keep_cols = [col for col in keep_cols if not col.startswith('meta_')]
        step_dict['protein_counts'][cell_line] = len(non_meta_keep_cols)
        
        if print_flag:
            print(f"[{cell_line}] {num_proteins_before} -> {num_proteins_after} prots")

        if graph_flag or verbose:
            threshold_holder.append(outlier_factor*mean_of_means)
            graphing_dict_before[cell_line] = col_means
            graphing_dict_after[cell_line] = col_means[keep_cols]

    
    if print_flag:
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
    
    if verbose:
        return {
            'filtered_data': filtered_data,
            'graphing_dict_before': graphing_dict_before,
            'graphing_dict_after': graphing_dict_after,
            #THIS LINE IS INCLUDED BECAUSE THE THRESHOLD IS CALCULATED INTERNALLY
            'threshold':threshold_holder,
            'step_dict': step_dict
        }
    return filtered_data, step_dict


#log transforming:
def log2_transform_by_control(data_by_cell_line, cell_lines, control_data_by_cell_line, **kwargs):
    """
    Log2 transform each protein column in data_by_cell_line by dividing by the corresponding control value,
    then taking the log2 of the result.
    """
    # Extract kwargs for consistency with other filtering functions
    print_flag = kwargs.pop('print_flag', False)
    graph_flag = kwargs.pop('graph_flag', False)
    graph_type = kwargs.pop('graph_type', 'hist')
    filter_flag = kwargs.pop('filter_flag', True)
    verbose = kwargs.pop('verbose', True)
    
    if not filter_flag:
        print('log2 transform by control is disabled')
        return data_by_cell_line, {}, {}, None
    
    print('log2 transforming by control')
    log2_transformed,graphing_dict_before, graphing_dict_after = {},{},{}
    
    # Initialize step dictionary
    step_dict = {'step_name': 'log2_trans', 'protein_counts': {}}
    
    for cell in cell_lines:
        df = data_by_cell_line[cell].copy()
        control_vals = control_data_by_cell_line[cell].copy()
        control_vals['meta_Inhi_5'] = 100
        control_vals['meta_Inhi_05'] = 100
        control_vals['meta_Inhi_50'] = 100
        control_vals['meta_Inhi_200'] = 100
        
        # Only transform columns that are present in both data and control, and are numeric
        protein_cols = list(set(df.select_dtypes(include='number').columns).intersection(set(control_vals.columns)))
        #log transforming
        for col in protein_cols:
            control_val = control_vals[col].values
            df[col] = np.log2(df[col] / control_val)

        log2_transformed[cell] = df
        # Count only non-meta proteins for the step dictionary
        non_meta_prots = len([col for col in df.columns if not col.startswith('meta_')])
        step_dict['protein_counts'][cell] = non_meta_prots

        if graph_flag or verbose:
            graphing_dict_before[cell] = data_by_cell_line[cell][protein_cols].copy()
            graphing_dict_after[cell] = log2_transformed[cell][protein_cols].copy()

        if print_flag:
            print(f"[{cell}] Log2 transformed {len(protein_cols)} protein columns")
    

    if graph_flag:
        # Plot before and after transformation if requested
        plot_grid_graphs(
            graphing_dict_before, 
            cell_lines, 
            title_func=lambda cell: f'{cell} - Before Log2 Transform',
            xlabel_func=lambda: 'Raw Expression Values',
            graph_type=graph_type,
            **kwargs
        )
        plot_grid_graphs(
            graphing_dict_after, 
            cell_lines, 
            title_func=lambda cell: f'{cell} - After Log2 Transform', 
            xlabel_func=lambda: 'Log2 Transformed Values',
            graph_type=graph_type,
            **kwargs
        )
        
    if print_flag:
        print('\n\n')
    if verbose:
        return {
            'filtered_data': log2_transformed,
            'graphing_dict_before': graphing_dict_before,
            'graphing_dict_after': graphing_dict_after,
            'step_dict': step_dict
        }
    return log2_transformed,


#complex filters
def filter_by_mutual_information(data_by_cell_line,cell_lines, mi_thresh=0.01,y_col=None,rand_state=42, **kwargs):
    filter_flag=kwargs.pop('filter_flag',True)
    if not filter_flag:
        print('filtering by mutual information is disabled')
        return data_by_cell_line,None

    filtered_data,graphing_dict_before,graphing_dict_after={},{},{}
    print_flag=kwargs.pop('print_flag',False)
    graph_flag=kwargs.pop('graph_flag',False)
    graph_type=kwargs.pop('graph_type','hist')
    verbose=kwargs.pop('verbose',True)
    print('filtering by mutual information')
    
    # Initialize step dictionary
    step_dict = {'step_name': 'mut_info', 'protein_counts': {}}



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
        
        # Store protein count for step dictionary
        step_dict['protein_counts'][cell] = len(keep_cols)
        
        if print_flag:
            print(f"[{cell}] {raw_data.shape[1]} -> {len(keep_cols)} prots")

        if graph_flag or verbose:
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
    if print_flag:
        print('\n\n')
    
    if verbose:
        return {
            'filtered_data': filtered_data,
            'graphing_dict_before': graphing_dict_before,
            'graphing_dict_after': graphing_dict_after,
            'step_dict': step_dict
        }
    return filtered_data, step_dict

def iterative_signal_filtering(data_by_cell_line,cell_lines, std_threshold=2.5, tole=0.001, filtering_to_use=1, **kwargs):
    #logic of the code: assume we are dealing with gaussian noise initially, and then if there are things that stand out from the noise, we pluck them out
    #then we repeat the process until it seems like we have plucked out all the signal and we are truly left with just noise, and then we return all the proteins
    #that have a single non-signal value
    print('filtering using iterative SNR filtering')
    print_flag=kwargs.pop('print_flag',False)
    graph_flag=kwargs.pop('graph_flag',False)
    graph_type=kwargs.pop('graph_type','hist')
    graphing_dict_before,graphing_dict_after,filtered_data={},{},{}
    filter_flag=kwargs.pop('filter_flag',True)
    verbose=kwargs.pop('verbose',True)
    if not filter_flag:
        print('filtering by iterative signal filtering is disabled')
        return data_by_cell_line, None
    
    # Initialize step dictionary
    step_dict = {'step_name': 'iter_signal', 'protein_counts': {}}


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
        
        # Store protein count for step dictionary
        step_dict['protein_counts'][cell] = filtered_df.shape[1]
        
        if print_flag:
            print(f"[{cell}] {n_start} -> {filtered_df.shape[1]} prots")

        if graph_flag or verbose:
            pass
            # print('graphing not implemented')
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
    if print_flag:
        print('\n\n')
    
    if verbose:
        return {
            'filtered_data': filtered_data,
            'graphing_dict_before': graphing_dict_before,
            'graphing_dict_after': graphing_dict_after,
            'step_dict': step_dict
        }
    return filtered_data, step_dict


#spearman and pearson filtering
def spearman_corr_filtering(data_by_cell_line, cell_lines, threshold=0.01, ycol='meta_Inhi_5', **kwargs):
    from scipy.stats import spearmanr
    filtered_data,graphing_dict_before,graphing_dict_after={},{},{}
    print_flag=kwargs.pop('print_flag',False)
    graph_flag=kwargs.pop('graph_flag',False)
    graph_type=kwargs.pop('graph_type','hist')
    filter_flag=kwargs.pop('filter_flag',True)
    verbose=kwargs.pop('verbose',True)
    if not filter_flag:
        print('filtering by spearman correlation is disabled')
        return data_by_cell_line, None
    
    print('filtering by spearman correlation')
    
    # Initialize step dictionary
    step_dict = {'step_name': 'spearman', 'protein_counts': {}}

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

        # Store protein count for step dictionary
        step_dict['protein_counts'][cell] = len(keep_cols)

        if print_flag:
            print(f"[{cell}] {cell_data.shape[1]} -> {len(keep_cols)} prots")

        if graph_flag or verbose:
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
    if print_flag:
        print('\n\n')
    
    if verbose:
        return {
            'filtered_data': filtered_data,
            'graphing_dict_before': graphing_dict_before,
            'graphing_dict_after': graphing_dict_after,
            'step_dict': step_dict
        }
    return filtered_data, step_dict

def pearson_corr_filtering(data_by_cell_line, cell_lines, threshold=0.01, ycol='meta_Inhi_5', **kwargs):
    from scipy.stats import pearsonr
    filtered_data,graphing_dict_before,graphing_dict_after={},{},{}
    print_flag=kwargs.pop('print_flag',False)
    graph_flag=kwargs.pop('graph_flag',False)
    graph_type=kwargs.pop('graph_type','hist')
    filter_flag=kwargs.pop('filter_flag',True)
    verbose=kwargs.pop('verbose',True)
    if not filter_flag:
        print('filtering by pearson correlation is disabled')
        return data_by_cell_line, None
    
    print('filtering by pearson correlation')
    
    # Initialize step dictionary
    step_dict = {'step_name': 'pearson', 'protein_counts': {}}

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

        # Store protein count for step dictionary
        step_dict['protein_counts'][cell] = len(keep_cols)

        if print_flag:
            print(f"[{cell}] {cell_data.shape[1]} -> {len(keep_cols)} prots")

        if graph_flag or verbose:
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
    if print_flag:
        print('\n\n')
    
    if verbose:
        return {
            'filtered_data': filtered_data,
            'graphing_dict_before': graphing_dict_before,
            'graphing_dict_after': graphing_dict_after,
            'step_dict': step_dict
        }
    return filtered_data, step_dict


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


#LOO regression:

def loo_regression_per_cell_line(
    data_by_cell_line,
    cell_lines,
    ycol,
    model,
    print_stats=True,
    plot_results=True,
    meta_regex='meta_',
    show_progress=True,
    hash_value='None',
    **kwargs
):
    """
    Perform Leave-One-Out regression for each cell line using the specified model.

    Parameters:
    - data_by_cell_line: dict of {cell_line: DataFrame}
    - cell_lines: list of cell line names to process
    - ycol: str, name of the target column (must be present in each DataFrame)
    - model: sklearn-like estimator instance
    - print_stats: bool, whether to print R2/MSE per cell line
    - plot_results: bool, whether to plot true vs predicted for each cell line
    - meta_regex: str, regex to identify meta columns to drop from X
    - show_progress: bool, whether to show progress bar during LOO iterations
    - **kwargs: plotting config (n_rows, n_cols, figsize_per_subplot, etc.)

    Returns:
    - loo_results: dict of {cell_line: {'r2':..., 'mse':..., 'pearson':..., 'y_true':..., 'y_pred':...}}
    """
    if cell_lines is None:
        cell_lines = list(data_by_cell_line.keys())

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

    if plot_results:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize_per_subplot[0] * n_cols, figsize_per_subplot[1] * n_rows))
        axes = axes.flatten()

    cell_line_iterator = tqdm(enumerate(cell_lines), total=len(cell_lines), desc="Processing cell lines", position=0)

    for i, cell_line in cell_line_iterator:
        df = data_by_cell_line[cell_line]
        meta_cols = df.filter(regex=meta_regex, axis=1)
        X = df.drop(columns=meta_cols.columns)

        if ycol not in df.columns:
            tqdm.write(f"[{cell_line}] Target column '{ycol}' not found. Skipping.")
            continue
        if X.empty:
            tqdm.write(f"[{cell_line}] Feature matrix X is empty. Skipping.")
            continue

        y = df[ycol]
        X = X.astype(float).fillna(0)
        y = y.astype(float).fillna(0)

        n_features = X.shape[1]
        loo = LeaveOneOut()
        y_true = []
        y_pred = []

        if show_progress:
            loo_iterator = tqdm(loo.split(X), total=X.shape[0], desc=f"LOO for {cell_line}", position=1, leave=False)
        else:
            loo_iterator = loo.split(X)

        for train_index, test_index in loo_iterator:
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            model_instance = clone(model)
            model_instance.fit(X_train, y_train)
            pred = model_instance.predict(X_test)
            y_true.append(y_test.values[0])
            y_pred.append(pred[0])

        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        pearson_corr = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else np.nan

        loo_results[cell_line] = {
            'r2': r2,
            'mse': mse,
            'pearson': pearson_corr,
            'y_true': y_true,
            'y_pred': y_pred
        }

        if print_stats:
            tqdm.write(
                f"[{cell_line}] LOO R2: {r2:.3f} | MSE: {mse:.3f} | Pearson: {pearson_corr:.3f} | Features: {n_features}"
            )

        if plot_results and i < len(axes):
            axes[i].scatter(y_true, y_pred, alpha=0.7)
            axes[i].plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label='y = y_pred')
            axes[i].set_xlabel('True Values')
            axes[i].set_ylabel('Predicted Values')
            axes[i].set_title(f'{cell_line}\nR2={r2:.2f}, MSE={mse:.2f}, ρ={pearson_corr:.2f}')
            axes[i].legend()

    if plot_results:
        for j in range(len(cell_lines), n_rows * n_cols):
            fig.delaxes(axes[j])
        fig.text(
            0.98, 0.98, f'Hash: {hash_value}',
            ha='right', va='top',
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )
        plt.tight_layout()
        plt.show()

    return loo_results


    

        











