"""
This is the main script which contains interactive model construction and training/testing
"""
import cellbox
import os
import numpy as np
import pandas as pd
import torch
import shutil
import argparse
import json


parser = argparse.ArgumentParser(description='CellBox main script')
parser.add_argument('-config', '--experiment_config_path', required=True, type=str, help="Path of experiment config")
parser.add_argument('-i', '--working_index', default=0, type=int)
parser.add_argument('-drug', '--drug_index', required=False, type=int)
parser.add_argument('-weights', '--weights', required=False, type=str, default=None)
parser.add_argument('--extra_mask_path', required=False, type=str, default=None,
                    help="Optional path to an extra adjacency mask (.npy/.npz/.csv/.tsv/.txt) to AND with base mask")
parser.add_argument('--extra_mask_none', action='store_true',
                    help="Explicitly disable any extra mask even if present in config")
master_args = parser.parse_args()
print('master_args')
print(master_args)
print('bye')

def set_seed(in_seed):
    """Sets random seeds for numpy and tensorflow."""
    int_seed = int(in_seed)
    torch.manual_seed(int_seed)
    np.random.seed(int_seed)


def prepare_workdir(in_cfg):
    """Creates the working directory for each experiment and generates necessary files."""
    # Read Data
    in_cfg.root_dir = os.getcwd()
    in_cfg.node_index = pd.read_csv(in_cfg.node_index_file, header=None, names=None) \
        if hasattr(in_cfg, 'node_index_file') else pd.DataFrame(np.arange(in_cfg.n_x))

    # Create Output Folder
    #this is path to results folder which has been modified to be in Abdullah_kuziez/Experiments/results
    experiment_path = os.path.join('Abdullah_kuziez', 'Experiments', 'results', '{}_{}'.format(in_cfg.experiment_id, md5))
    try:
        os.makedirs(experiment_path)
    except Exception:
        pass
    out_cfg = vars(in_cfg)
    out_cfg = {key: out_cfg[key] for key in out_cfg if type(out_cfg[key]) is not pd.DataFrame}
    os.chdir(experiment_path)
    json.dump(out_cfg, open('config.json', 'w'), indent=4)
    # Write config md5 hash to a separate text file to keep JSON strict
    with open('config_md5.txt', 'w') as f_md5:
        f_md5.write(md5 + '\n')

    if "leave one out" in in_cfg.experiment_type:
        try:
            in_cfg.model_prefix = '{}_{}'.format(in_cfg.model_prefix, in_cfg.drug_index)
        except Exception('Drug index not specified') as e:
            raise e

    in_cfg.working_index = in_cfg.model_prefix + "_" + str(working_index).zfill(3)

    try:
        shutil.rmtree(in_cfg.working_index)
    except Exception:
        pass
    os.makedirs(in_cfg.working_index)
    os.chdir(in_cfg.working_index)

    with open("record_eval.csv", 'w') as f:
        f.write("epoch,iter,train_loss,valid_loss,train_mse,valid_mse,test_mse,time_elapsed\n")

    print('Working directory is ready at {}.'.format(experiment_path))
    return 0


if __name__ == '__main__':
    # Launching expr
    working_index = master_args.working_index
    cfg = cellbox.config.Config(master_args.experiment_config_path)
    cfg.ckpt_path_full = os.path.join('./', cfg.ckpt_name)
    cfg.weights = master_args.weights
    # Extra mask wiring: CLI overrides config
    if getattr(master_args, 'extra_mask_none', False):
        cfg.extra_mask_path = None
        cfg.extra_mask = None
    elif hasattr(master_args, 'extra_mask_path') and master_args.extra_mask_path not in [None, "None", ""]:
        cfg.extra_mask_path = master_args.extra_mask_path
    md5 = cellbox.utils_torch.md5(cfg)
    cfg.drug_index = master_args.drug_index if hasattr(master_args, "drug_index") else None
    cfg.seed = working_index + cfg.seed if hasattr(cfg, "seed") else working_index + 1000
    set_seed(cfg.seed)
    print(vars(cfg))

    prepare_workdir(cfg)
    logger = cellbox.utils_torch.TimeLogger(time_logger_step=1, hierachy=3)
    args = cfg
    for i, stage in enumerate(cfg.stages):
        set_seed(cfg.seed)
        cfg = cellbox.dataset_torch.factory(cfg)
        logger.log("Training on stage {}/{} ...".format(i + 1, len(cfg.stages)))
        args.sub_stages = stage['sub_stages']
        args.n_T = stage['nT']
        model = cellbox.model_torch.factory(args)
        cellbox.train_torch.train_model(model, args)
