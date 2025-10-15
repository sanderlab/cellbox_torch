"""
This script is used to process the results of the cellbox_torch model. It will be used to plot results for the graphs
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import anndata as ad
from scipy.stats import zscore
sns.set_theme()
from matplotlib.lines import Line2D
from scipy.stats import pearsonr
import json

class results_holder():
    def __init__(self, config_path, hash_path):
        self.config_path = config_path
        self.hash=hash_path
        self.loss_path = None
        self.expr_path = None
        self.yhat_path = None
        self.row_matching_path = None
    

    def get_paths(self):
        """This fxn pulls the names for the important results files from the folder"""
        os.chdir(os.path.dirname(self.config_path))
        os.chdir('..')
        os.chdir('..')
        #read in the config file using json
        with open(self.config_path, 'r') as file:
            config_dict = json.load(file)
        os.chdir('..')

        #pull out the values of the data used for the experiment
        expr_path = config_dict['expr_file']
        expr_path = os.path.abspath(expr_path)
        os.chdir('Abdullah_kuziez')
        exp_title=config_dict['experiment_id']

        #grab the paths using string patterns
        # go to results and check for a folder that contains the string pattern of experiment id
        results_dir = os.path.join(os.getcwd(), "Experiments", "results")
        # Find the result folder that contains the experiment id
        exp_folder = None
        for folder in os.listdir(results_dir):
            if exp_title in folder:
                exp_folder = os.path.join(results_dir, folder)
                break
        if exp_folder is None:
            raise FileNotFoundError(f"No results folder found containing experiment id: {exp_title}")

        # Find the seed folder (e.g., seed_000)
        seed_folder = None
        for folder in os.listdir(exp_folder):
            if folder.startswith("seed_"):
                seed_folder = os.path.join(exp_folder, folder)
                break
        if seed_folder is None:
            raise FileNotFoundError(f"No seed folder found in {exp_folder}")

        # Compose the paths
        # Find the yhat file with the highest leading number and containing "best.y_hat.loss"
        import re
        yhat_files = []
        for file in os.listdir(seed_folder):
            match = re.match(r"(\d+)_best\.y_hat\.loss", file)
            if match:
                leading_num = int(match.group(1))
                yhat_files.append((leading_num, file))
        if not yhat_files:
            raise FileNotFoundError("No yhat file found containing '_best.y_hat.loss' in seed folder.")
        # Pick the file with the highest leading number
        yhat_files.sort(reverse=True)
        yhat_path = os.path.join(seed_folder, yhat_files[0][1])

        row_matching_path = os.path.join(seed_folder, "random_pos.csv")
        loss_path = os.path.join(seed_folder, "record_eval.csv")


        self.loss_path = loss_path
        self.expr_path = expr_path
        self.yhat_path = yhat_path
        self.row_matching_path = row_matching_path
        return loss_path, expr_path, yhat_path, row_matching_path


    def plot_train_valid_loss(loss_rel_path):
        losses = pd.read_csv(loss_rel_path, index_col=False)
        train_loss = pd.to_numeric(losses['train_loss'], errors='coerce')
        valid_loss = pd.to_numeric(losses['valid_loss'], errors='coerce')
        plt.figure(figsize=(8,5))
        plt.plot(train_loss, label='Train Loss')
        plt.plot(valid_loss, label='Validation Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.tight_layout()

    def plot_train_valid_mse(loss_rel_path):
        losses = pd.read_csv(loss_rel_path, index_col=False)
        train_mse = pd.to_numeric(losses['train_mse'], errors='coerce')
        valid_mse = pd.to_numeric(losses['valid_mse'], errors='coerce')
        plt.figure(figsize=(8,5))
        plt.plot(train_mse, label='Train MSE')
        plt.plot(valid_mse, label='Validation MSE')
        plt.xlabel('Iteration')
        plt.ylabel('MSE')
        plt.title("Training and Validation MSE")
        plt.legend()
        plt.tight_layout()

    def plot_loss_curves(loss_rel_path):
        self.plot_train_valid_loss(loss_rel_path)
        plt.show()
        self.plot_train_valid_mse(loss_rel_path)
        plt.show()



def plot_pearson_correlation(self,col_numbers_pheno=-5):
    y = pd.read_csv(self.expr_path, header=None)
    yhat = pd.read_csv(self.yhat_path).iloc[:,1:]
    row_matching = pd.read_csv(self.row_matching_path, header=None)

    # Get test indices (assuming test set is at the end)
    test_indices = row_matching[-(yhat.shape[0]):]

    test_indices_list = test_indices[0].values

    #phenotype stuff got bastardized: fixed temporarily
    # if col_numbers_pheno==-5:
    #     col_numbers_pheno=yhat.columns.get_loc('Cell_viability%_(cck8Drug-blk)/(control-blk)*100')
    # Get all columns except the phenotype column
    all_columns = list(range(yhat.shape[1]))
    prot_columns = [col for col in all_columns if col != col_numbers_pheno]


    y_true_pheno=y.iloc[test_indices_list,col_numbers_pheno].to_numpy().flatten().astype(float)
    y_true_prots=y.iloc[test_indices_list,prot_columns].to_numpy().flatten().astype(float)
    yhat_pheno=yhat.iloc[:,col_numbers_pheno].to_numpy().flatten().astype(float)
    yhat_prots=yhat.iloc[:,prot_columns].to_numpy().flatten().astype(float)

    plt.figure(figsize=(8, 8))

    # Plot molecular nodes: light blue, slightly transparent, small dots
    plt.scatter(
        y_true_prots, yhat_prots,
        label='Proteins',
        color='#6baed6',  # light blue
        alpha=0.5,
        s=12
    )

        # Plot phenotype nodes: dark blue, slightly transparent, small dots
    plt.scatter(
        y_true_pheno, yhat_pheno,
        label='Phenotype',
        color='#08306b',  # dark blue
        alpha=0.7,
        s=12
    )

    # Plot y=x line
    all_true = np.concatenate((y_true_pheno, y_true_prots))
    all_pred = np.concatenate((yhat_pheno, yhat_prots))
    min_val = min(np.min(all_true), np.min(all_pred))
    max_val = max(np.max(all_true), np.max(all_pred))
    plt.plot([min_val, max_val], [min_val, max_val], color='gray', linestyle='--', linewidth=1, label='y = x')
    # Calculate Pearson's r for phenotype and proteins
    r_pheno, _ = pearsonr(y_true_pheno, yhat_pheno)
    r_prots, _ = pearsonr(y_true_prots, yhat_prots)
    # Write on the graph
    # Calculate single Pearson's r for all points
    r_all, _ = pearsonr(all_true, all_pred)
    plt.text(
        0.95, 0.05,
        f"Pearson r: {r_all:.3f}",
        transform=plt.gca().transAxes,
        fontsize=11,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
    )

    plt.legend()
    plt.xlabel('True Expression')
    plt.ylabel('Predicted Expression')

    plt.title('Correlation between experiment and prediction in test set:')
    plt.tight_layout()
    return