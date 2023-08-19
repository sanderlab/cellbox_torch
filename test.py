import pytest
import os
import glob
import torch
import tensorflow as tf
import numpy as np

from test_utils.test_cases import KernelConfig, \
    ODE_ERROR_TOLERANCE, ODE_PARAMS, ODE_TEST_CASES, ODE_GROUND_TRUTHS

#from test_utils.dataloader import get_dataloader, yield_data_from_tensorflow_dataloader, yield_data_from_pytorch_dataloader, \
#    s2c_row_inds, loo_row_inds


#def test_model():
#    os.system('python scripts/main.py -config=configs/Example.minimal.json')
#    files = glob.glob('results/Debugging_*/seed_000/3_best.W*')
#    assert False

#################################################### Tests for kernel functions ####################################################

# Test for ODE solver
def test_ode_solver():
    """
    Test Tensorflow's and Pytorch's kernel functions with multiple input test cases
    """
    ode_args = ODE_TEST_CASES[0]
    ode_ground_truths = ODE_GROUND_TRUTHS[0]
    ode_params = ODE_PARAMS
    # Manually determine the params and all other necessary params
    #args = KernelConfig()
    #W = np.random.randn(args.n_x, args.n_x)
    #eps = np.ones((args.n_x, 1), dtype=np.float32)
    #alpha = np.ones((args.n_x, 1), dtype=np.float32)
    #y0_np = np.zeros((args.n_x, 1))
    #gradient_zero_from = None
    #mu_t_np = np.random.randn(args.n_x, 4)
#
    ### Tensorflow's code
    ## import the relevant objects
    #from cellbox.kernel import get_ode_solver, get_envelope, get_dxdt
#
    #params = {}
    #W_copy = np.copy(W)
    #params["W"] = tf.convert_to_tensor(W_copy, dtype=tf.float32)
    #params["alpha"] = tf.convert_to_tensor(np.log(np.exp(alpha) + 1), dtype=tf.float32)
    #params["eps"] = tf.convert_to_tensor(np.log(np.exp(eps) + 1), dtype=tf.float32)
    #envelope_fn = get_envelope(args)
    #ode_solver = get_ode_solver(args)
    #_dxdt = get_dxdt(args, params)

    # Determine the input and output shape of the ODE
    #mu_t_copy = np.copy(mu_t_np)
    #y0 = tf.convert_to_tensor(y0_np, dtype=tf.float32)
    #mu_t = tf.convert_to_tensor(mu_t_copy, dtype=tf.float32)
    #ys_tf = ode_solver(y0, mu_t, args.dT, args.n_T, _dxdt, gradient_zero_from)
    #ys_tf = ys_tf.eval(session=tf.compat.v1.Session())

    ## Pytorch's code
    # import the relevant objects
    #from cellbox.kernel_torch import get_ode_solver, get_envelope, get_dxdt

    params = {}
    params["W"] = torch.tensor(ode_params["W"], dtype=torch.float32)
    params["alpha"] = torch.tensor(np.log(np.exp(ode_params["alpha"]) + 1), dtype=torch.float32)
    params["eps"] = torch.tensor(np.log(np.exp(ode_params["eps"]) + 1), dtype=torch.float32)
    envelope_fn = get_envelope(args)
    ode_solver = get_ode_solver(args)
    _dxdt = get_dxdt(args, params)

    # Determine the input and output shape of the ODE
    y0 = torch.tensor(ode_params["y0_np"], dtype=torch.float32)
    mu_t = torch.tensor(ode_params["mu_t_np"], dtype=torch.float32)
    ys_torch = ode_solver(y0, mu_t, args.dT, args.n_T, _dxdt, args.gradient_zero_from).detach().numpy()

    # Determine an error tolerance
    mean_diff = np.mean(np.abs(ys_torch - ode_ground_truths))
    median_diff = np.median(np.abs(ys_torch - ode_ground_truths))
    assert mean_diff < ODE_ERROR_TOLERANCE
    assert median_diff < ODE_ERROR_TOLERANCE



               

if __name__ == '__main__':

    pytest.main(args=['-sv', os.path.abspath(__file__)])
