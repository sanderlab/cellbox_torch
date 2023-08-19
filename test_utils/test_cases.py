# This contains test cases for each test
import numpy as np
import torch
import glob

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