# This contains test cases for each test

#################################### KERNELS ####################################
class KernelConfig(object):
    def __init__(self):
        
        self.n_x = 99
        self.envelope_form = "tanh" 
        self.envelope_fn = None
        self.polynormial_k = None
        self.ode_degree = 1
        self.envelope = 0
        self.ode_solver = "heun"
        self.dT = 0.1
        self.n_T = 200
        self.gradient_zero_from = None
        #self.__dict__.update(config_dict)

ODE_ERROR_TOLERANCE = 0.01



a = {
  "experiment_id": "Example_RP",
  "experiment_type": "random partition",
  "model": "CellBox",
  "sparse_data": False,
  "pert_file": "/users/ngun7t/Documents/cellbox-jun-6/data/pert.csv",
  "expr_file": "/users/ngun7t/Documents/cellbox-jun-6/data/expr.csv",
  "node_index_file": "/users/ngun7t/Documents/cellbox-jun-6/data/node_Index.csv",
  "n_protein_nodes": 82,
  "n_activity_nodes": 87,
  "n_x" : 99,
  "trainset_ratio": 0.7,
  "validset_ratio": 0.8,
  "batchsize": 4,
  "add_noise_level": 0,

  "envelop_form": "tanh",
  "dT": 0.1,
  "envelop":0,
  "ode_degree": 1,
  "ode_solver": "heun",
  "ode_last_steps": 2,
  "l1lambda": 1e-4,
  "l2lambda": 1e-4,

  "n_epoch": 100,
  "n_iter": 100,
  "n_iter_buffer":50,
  "n_iter_patience":100,

  "stages":[{
    "nT": 200,
    "sub_stages":[
        {"lr_val": 0.001,"l1lambda": 0.0001}
    ]}],

    "export_verbose": 3,
    "ckpt_name": "model11.ckpt"
}