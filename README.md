<a target="_blank" href="https://colab.research.google.com/drive/1BadFag4PnxyLFaeu9QZuvQGvYHBfYgEt?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


# CellBox pytorch
This is a Pytorch implementation of CellBox, a model for cell perturbation biology. The original version was written in Tensorflow 1 and contains many outdated functions. This implementation is written in Pytorch 2.0 and has been extensively test and verified to work similarly to the original version. 


## Abstract
Systematic perturbation of cells followed by comprehensive measurements of molecular and phenotypic responses provides informative data resources for constructing computational models of cell biology. Models that generalize well beyond training data can be used to identify combinatorial perturbations of potential therapeutic interest. Major challenges for machine learning on large biological datasets are to find global optima in a complex multi-dimensional space and mechanistically interpret the solutions. To address these challenges, we introduce a hybrid approach that combines explicit mathematical models of cell dynamics with a machine learning framework, implemented in TensorFlow. We tested the modeling framework on a perturbation-response dataset of a melanoma cell line after drug treatments. The models can be efficiently trained to describe cellular behavior accurately. Even though completely data-driven and independent of prior knowledge, the resulting de novo network models recapitulate some known interactions. The approach is readily applicable to various kinetic models of cell biology.

<p align="center">
	<img src="https://lh3.googleusercontent.com/d/15Lildcx8sC4shTalODLXqfibJTbnxmun=w600">
</p>

## Citation and Correspondence

This CellBox pytorch implementation is based on the original CellBox scripts developed in Sander lab for the paper in _[Cell Systems](https://www.cell.com/cell-systems/pdfExtended/S2405-4712(20)30464-6)_ or [bioRxiv](https://www.biorxiv.org/content/10.1101/746842v3) maintained by Bo Yuan, Judy Shen, and Augustin Luna.

>Yuan, B.*, Shen, C.*, Luna, A., Korkut, A., Marks, D., Ingraham, J., Sander, C. CellBox: Interpretable Machine Learning for Perturbation Biology with Application to the Design of Cancer Combination Therapy. _Cell Systems_, 2020. 

This specific pytorch implementation is maintained by Phuc Nguyen, Augustin Luna, and Bo Yuan.

If you want to discuss the usage or to report a bug, please use the 'Issues' function here on GitHub.

If you find `CellBox` useful for your research, please consider citing the corresponding publication.

For more information, please find our contact information [here](https://www.sanderlab.org/#/). 

# Quick Start with Colab

Easily try `CellBox pytorch` online with Google Colab. Go to https://colab.research.google.com/drive/1BadFag4PnxyLFaeu9QZuvQGvYHBfYgEt?usp=sharing and run the notebook.

# Installation

## Install using pip 
Before installing CellBox-pytorch, it is good practice to create a Python virtual environment. With conda, `conda create -n cellbox python==3.8.0` creates a conda environment with the name `cellbox` and Python 3.8.0. Activate the environment by `conda activate cellbox`. 

To install CellBox-pytorch to a particular folder, type the following:

```
git clone https://github.com/sanderlab/CellBox.git <folder_name>
cd /<folder_name>/cellbox
pip install .
```


# Project Structure

## ./data/ folder in GitHub repo
These data files are used for generating the results from the official CellBox paper. **However**, the CellBox model does not include a data preprocessing pipeline that converts your data into CellBox-compatible format. The CellBox data was first extracted from this paper, and several steps of data normalization were done before the final perturbation matrix was obtained. CellBox-pytorch and the original CellBox repo have been designed currently to only work with CellBox data files, and current efforts to run CellBox on other forms of data are undergoing.
* `node_index.csv`: names of each protein/phenotypic node.
* `expr_index.txt`: information each perturbation condition. This is one of the original data files we downloaded from [paper](https://elifesciences.org/articles/04640) and is only used here as a reference for the condition names. In other words the 2nd and 3rd columns are not being used in CellBox.
* `loo_label.csv`: A deprecated csv file that stores the actual indexing of perturbation targets, used in the original paper. There are 89 rows corresponding to 89 drug combinations. On each row, two numbers denote the index of one of 12 drugs for that combination. Number 0 denotes no drug, meaning rows with 0 denote single-target drugs.
* `expr.csv`: Protein expression data from RPPA for the protein nodes and phenotypic node values. Each row is a condition while each column is a node.
* `pert.csv`: Perturbation strength and target of all perturbation conditions. Used as input for differential equations.
* `expr_subset.npz` and `pert_subset.npz`: A subset of `expr.csv` and `pert.csv` (deprecated).

## CellBox-pytorch package main components:
* The main structure of `CellBox-pytorch` model is defined in `model_torch.py`.
* A `dataset_torch.factory()` function for random parition, leave-one-out, and single-to-combo tasks (refer to the original CellBox paper for more information).
* A multiple-substage training process for finding the optimal hyperparameters defined in `train_torch.py`. 

## Model construction and training

### __Step 1: Create experiment json files (some examples can be found under ./configs/)__
* Make sure to specify the experiment_id and experiment_type
	* `experiment_id`: name of the experiments, used to generate a result folder in a format of `<experiment_id>_<random_string>`, where `random_string` is unique for each config file. When training CellBox-pytorch using the same config file but different seeds, each seed result is stored as a subfolder to `<experiment_id>_<random_string>`.
	* `experiment_type`: currently available tasks are {"random partition", "leave one out (w/o single)", "leave one out (w/ single)", "full data", "single to combo"}. This refers to different methods for partitioning the original perturbation matrix.
* Different training stages can be specified using `stages` and `sub_stages` in config file. Each `stage` is a list containing information about `sub_stages`, which are grouped based on the ODE time steps (`nT`). Each `sub_stage` within a same group differs from each other on the learning rate `lr_val` and l1-regularization coefficient `l1_lambda`, with later `sub_stages` having incrementally smaller `lr_val` and `l1_lambda`.
* Other default configurations are defined in `config.py`

### __Step 2: Train CellBox-pytorch__

To start training the model, run `python scripts/main.py`. **Note: always run the script in the root folder**. 

The following are the arguments for `python scripts/main.py`:
* `--experiment_config_path` or `-config` (required): The path to the experiment type configuration file. 
* `--working_index` or `-i` (optional): A random seed for random partitioning and CellBox weight initialization. More information on this option below.
* `--drug_index` or `-drug` (required when `experiment_type` is `leave one out (w/o single)` and `leave one out (w/ single)`): An index to denote which drug is left out for testing.

For example:
* Running CellBox with random partition:

```
python scripts/main_torch.py -config=configs/Example.random_partition.json
```
* Setting a specific seed:

```
python scripts/main_torch.py -config=configs/Example.random_partition.json -i=1234
```

* Setting a specific drug index for leave-one-out experiments:
```
python scripts/main_torch.py -config=configs/Example.leave_one_drug_out.json -i=1234 -drug 5
```


### __Step 3: Analyze result files__
* You should see a experiment folder generated under `/results` using the date and `experiment_id`.
* Under experiment folder, you would see different models run with different random seeds
* Under each model folder, you would have:
	* `record_eval.csv`: log file with loss changes and time used.
	* `random_pos.csv` (only for random partitions): the data splits for training, validation, and testing. For example, train-val-test splits are 50-30-20, then the first 50% of rows in `random_pos.csv` files correspond to indices in the training set, the next 30% and 20% of rows correspond to validation and test sets.
	* `best.W`, `best.alpha`, `best.eps`: model parameters snapshot for each training stage.
	* `best.y_hat`: Prediction on test set, using the best model for each stage. The loss value in the file name denotes the total loss (MSE + L1 loss + L2 loss) with that prediction. The rows of the file correspond to the test indices in `random_pos.csv` if random partitioning, or the left out drug in other experiment tasks.
	* `.pth` files are the final models in pytorch compatible format.
	* `best.summary`: Prediction on train, val, and test sets, using the best model for each stage.
	
	
# Technical discussions

## Unit tests for future development

Along with a new CellBox-pytorch implementation, this repo also contains unit tests for future verification efforts, implemented in `test_torch.py`, `/test_utils` and `/test_arrays`. Especially, `/test_arrays` include data obtained from the original CellBox and serve as a ground truth for further testing and development.