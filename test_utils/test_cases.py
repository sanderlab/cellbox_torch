# This contains test cases for each test
import numpy as np
import torch
import glob
import pickle
import pandas as pd
import os

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

#################################### ODE TEST CASES ####################################
class KernelConfig(object):
    def __init__(self, n_x, envelope_form, ode_solver, n_T):
        
        self.n_x = n_x
        self.envelope_form = envelope_form
        self.envelope_fn = None
        self.polynomial_k = 2
        self.ode_degree = 1
        self.envelope = 0
        self.ode_solver = ode_solver
        self.dT = 0.1
        self.ode_last_steps = 2
        self.n_T = n_T
        self.gradient_zero_from = None
        #self.__dict__.update(config_dict)


class ODETestCase(object):

    def __init__(self):

        with open("test_arrays/ODE_W.npy", "rb") as f:
            W = np.load(f)
        with open("test_arrays/ODE_mu_t_np.npy", "rb") as f:
            mu_t_np = np.load(f)

        self.SEED = 42
        self.N_X = 10
        self.ODE_PARAMS = {
            "W": W,
            "eps": np.ones((self.N_X, 1), dtype=np.float32),
            "alpha": np.ones((self.N_X, 1), dtype=np.float32),
            "y0_np": np.zeros((self.N_X, 4)),
            "mu_t_np": mu_t_np,
        }

        self.ODE_ERROR_TOLERANCE = 0.01
        self.ODE_TEST_CASES = [
            KernelConfig(self.N_X, "tanh", "heun", 100),
            KernelConfig(self.N_X, "tanh", "euler", 100),
            KernelConfig(self.N_X, "tanh", "rk4", 100),
            KernelConfig(self.N_X, "tanh", "midpoint", 100),
            KernelConfig(self.N_X, "clip linear", "heun", 100),
            KernelConfig(self.N_X, "clip linear", "euler", 100),
            KernelConfig(self.N_X, "clip linear", "rk4", 100),
            KernelConfig(self.N_X, "clip linear", "midpoint", 100),
        ]

        self.ODE_GROUND_TRUTHS = []
        for args in self.ODE_TEST_CASES:
            n = f"test_arrays/ODE_gt_{args.envelope_form}_{args.polynomial_k}_{args.ode_degree}_{args.envelope}_{args.ode_solver}_{args.dT}_{args.ode_last_steps}_{args.n_T}.npy"
            with open(n, "rb") as f:
                self.ODE_GROUND_TRUTHS.append(np.load(f))

        self.parametrized_tests = [
            (args, gt) for args, gt in zip(self.ODE_TEST_CASES, self.ODE_GROUND_TRUTHS)
        ]

ODE_PARAMETRIZED_TESTS = ODETestCase().parametrized_tests



#################################### DATALOADER TEST CASES ####################################
class DataloaderConfig(object):
    def __init__(self, experiment_type, drug_index):
        self.experiment_type = experiment_type
        self.sparse_data = False
        self.pert_file = "data/pert.csv"
        self.expr_file = "data/expr.csv"
        self.node_index_file = "data/node_Index.csv"
        self.root_dir = ""
        self.node_index = pd.read_csv(os.path.join(self.root_dir, self.node_index_file), header=None, names=None)
        self.n_protein_nodes = 82
        self.n_activity_nodes = 87
        self.n_x = 99
        self.trainset_ratio = 0.7
        self.validset_ratio = 0.8
        self.batchsize = 4
        self.add_noise_level = 0
        self.drug_index = drug_index

        self.l1lambda = 1e-4
        self.l2lambda = 1e-4
    
class DataloaderTestCase(object):

    def __init__(self, seed, exp_type, exp_type_abbrev, drug_index):
        self.seed = seed
        self.exp_type = exp_type
        self.exp_type_abbrev = exp_type_abbrev
        self.drug_index = drug_index
        self.cfg = DataloaderConfig(self.exp_type, self.drug_index)
        with open(f"test_arrays/dataloader/dataloader_{self.exp_type_abbrev}_{self.seed}_{self.drug_index}.pkl", "rb") as f:
            self.ground_truth = pickle.load(f)

DATALOADER_SEEDS = [940, 671, 601, 323, 259]
DATALOADER_EXPERIMENT_TYPES = [
    "random partition", 
    "leave one out (w/ single)", 
    "leave one out (w/o single)", 
    "single to combo", 
    "random partition with replicates"
]
DATALOADER_EXPERIMENT_TYPES_ABBREV = [
    "RP",
    "LOO-WS",
    "LOO-WoS",
    "S2C",
    "RPwRep"
]
DATALOADER_DRUG_INDICES = [2, 4, 5, 10, 11]
DATALOADER_PARAMETRIZED_TESTS = []
for seed, drug in zip(DATALOADER_SEEDS, DATALOADER_DRUG_INDICES):
    for exp, exp_abbrev in zip(DATALOADER_EXPERIMENT_TYPES, DATALOADER_EXPERIMENT_TYPES_ABBREV):
        DATALOADER_PARAMETRIZED_TESTS.append(
            DataloaderTestCase(seed, exp, exp_abbrev, drug)
        )


#################################### UTIL FUNCTIONS TEST CASES ####################################

class LossFunctionTestCase(object):

    def __init__(self, seed, mode, l1, l2, total_loss_tor=0.001, loss_mse_tor=10e-5):
        self.seed = seed
        self.mode = mode
        self.l1 = l1
        self.l2 = l2
        self.total_loss_tor = total_loss_tor
        self.loss_mse_tor = loss_mse_tor
        with open(f"test_arrays/loss_fn/loss-fn-input_{self.seed}_{self.l1}_{self.l2}_{self.mode}.pkl", "rb") as f:
            self.inp = pickle.load(f)
        with open(f"test_arrays/loss_fn/loss-fn-output_{self.seed}_{self.l1}_{self.l2}_{self.mode}.pkl", "rb") as f:
            self.out = pickle.load(f)
        
        self.x_gold = self.inp["x_gold"]
        self.x_hat = self.inp["x_hat"]
        self.W = self.inp["W"]
        self.total_loss = self.out["total_loss"]
        self.loss_mse = self.out["loss_mse"]

LOSS_L1 = [2.0, 0.1, 0.01]
LOSS_L2 = [3.0, 0.5, 0.05]
LOSS_SEEDS = [52, 16, 79]
LOSS_MODES = ["expr", "None"]
LOSS_PARAMETRIZED_TESTS = []
for mode in LOSS_MODES:
    for i, seed in enumerate(LOSS_SEEDS):
        LOSS_PARAMETRIZED_TESTS.append(
            LossFunctionTestCase(seed, mode, LOSS_L1[i], LOSS_L2[i])
        )


#################################### FEEDFORWARD TEST CASES ####################################

class ModelConfig(object):

    def __init__(self, model, n_x, envelope_form, ode_solver, n_T, pert_form):
        self.model = model
        self.n_x = n_x
        self.iter_train, self.iter_monitor, self.iter_eval = None, None, None
        self.lr = 0.1
        self.n_protein_nodes, self.n_activity_nodes = 82, 87
        self.pert_form = pert_form
        self.weights = "None"

        self.envelope_form = envelope_form
        self.envelope_fn = None
        self.polynomial_k = 2
        self.ode_degree = 1
        self.envelope = 0
        self.ode_solver = ode_solver
        self.dT = 0.1
        self.ode_last_steps = 2
        self.n_T = n_T
        self.gradient_zero_from = None


class FeedforwardTestCase(object):

    def __init__(self, args, seed, l1_lambda, l2_lambda, tolerance):
        self.args = args
        self.seed = seed
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.tolerance = tolerance

        with open(f"test_arrays/forward_pass/forward_input_{self.seed}_{self.l1_lambda}_{self.l2_lambda}.pkl", "rb") as f:
            self.data_in = pickle.load(f)
        with open(f"test_arrays/forward_pass/forward_out_{self.seed}_{self.l1_lambda}_{self.l2_lambda}.pkl", "rb") as f:
            self.data_out = pickle.load(f)


FEEDFORWARD_TOLERANCE = {
    "loss_total": 10e-5,
    "loss_mse": 10e-2,
    "yhat": 0.1
}
FEEDFORWARD_MODEL = "CellBox"
FEEDFORWARD_NX = 99
FEEDFORWARD_SEEDS = [7, 87, 62, 45, 23]
# In form of (l1_lambda, l2_lambda)
FEEDFORWARD_LAMBDAS = [(2.0, 3.0), (0.1, 0.01), (0.001, 0.0001), (0.01, 0.1), (0.0001, 0.001)]
# In form of (envelope_fn, ode_solver, "nT", "pert_form")
FEEDFORWARD_OTHERS = [
    ("tanh", "heun", 100, "by u"),
    ("tanh", "euler", 100, "by u"),
    ("clip linear", "heun", 100, "fix x"),
    ("tanh", "heun", 100, "fix x"),
    ("tanh", "midpoint", 100, "fix x")
]
FEEDFORWARD_PARAMETRIZED_TESTS = []
for seed, lamb, other in zip(FEEDFORWARD_SEEDS, FEEDFORWARD_LAMBDAS, FEEDFORWARD_OTHERS):
    FEEDFORWARD_PARAMETRIZED_TESTS.append(
        FeedforwardTestCase(
            ModelConfig(
                FEEDFORWARD_MODEL, 
                FEEDFORWARD_NX,
                other[0], other[1], other[2], other[3]
            ), seed, lamb[0], lamb[1], FEEDFORWARD_TOLERANCE
        )
    )

#a = {
#  "experiment_id": "Example_RP",
#  "experiment_type": "random partition",
#  "model": "CellBox",
#  "sparse_data": False,
#  "pert_file": "/users/ngun7t/Documents/cellbox-jun-6/data/pert.csv",
#  "expr_file": "/users/ngun7t/Documents/cellbox-jun-6/data/expr.csv",
#  "node_index_file": "/users/ngun7t/Documents/cellbox-jun-6/data/node_Index.csv",
#  "n_protein_nodes": 82,
#  "n_activity_nodes": 87,
#  "n_x" : 99,
#  "trainset_ratio": 0.7,
#  "validset_ratio": 0.8,
#  "batchsize": 4,
#  "add_noise_level": 0,
#
#  "envelop_form": "tanh",
#  "dT": 0.1,
#  "envelop":0,
#  "ode_degree": 1,
#  "ode_solver": "heun",
#  "ode_last_steps": 2,
#  "l1lambda": 1e-4,
#  "l2lambda": 1e-4,
#
#  "n_epoch": 100,
#  "n_iter": 100,
#  "n_iter_buffer":50,
#  "n_iter_patience":100,
#
#  "stages":[{
#    "nT": 200,
#    "sub_stages":[
#        {"lr_val": 0.001,"l1lambda": 0.0001}
#    ]}],
#
#    "export_verbose": 3,
#    "ckpt_name": "model11.ckpt"
#}