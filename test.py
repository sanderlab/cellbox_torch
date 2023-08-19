import pytest
import os
import glob
import torch
import tensorflow as tf
import numpy as np

from cellbox.kernel_torch import get_ode_solver, get_envelope, get_dxdt

from test_utils.test_cases import KernelConfig, \
    ODETestCase, ODE_PARAMETRIZED_TESTS

#from test_utils.dataloader import get_dataloader, yield_data_from_tensorflow_dataloader, yield_data_from_pytorch_dataloader, \
#    s2c_row_inds, loo_row_inds


#def test_model():
#    os.system('python scripts/main.py -config=configs/Example.minimal.json')
#    files = glob.glob('results/Debugging_*/seed_000/3_best.W*')
#    assert False

#################################################### Tests for kernel functions ####################################################

# Test for ODE solver
@pytest.mark.parametrize("args,ode_ground_truth", ODE_PARAMETRIZED_TESTS)
def test_ode_solver(args, ode_ground_truth):
    """
    Test Tensorflow's and Pytorch's kernel functions with multiple input test cases
    """
    ode_test_case = ODETestCase()
    np.random.seed(ode_test_case.SEED)
    ode_params = ode_test_case.ODE_PARAMS

    params = {}
    params["W"] = torch.tensor(np.copy(ode_params["W"]), dtype=torch.float32)
    params["alpha"] = torch.tensor(np.log(np.exp(ode_params["alpha"]) + 1), dtype=torch.float32)
    params["eps"] = torch.tensor(np.log(np.exp(ode_params["eps"]) + 1), dtype=torch.float32)
    envelope_fn = get_envelope(args)
    ode_solver = get_ode_solver(args)
    _dxdt = get_dxdt(args, params)

    # Determine the input and output shape of the ODE
    y0 = torch.tensor(np.copy(ode_params["y0_np"]), dtype=torch.float32)
    mu_t = torch.tensor(np.copy(ode_params["mu_t_np"]), dtype=torch.float32)
    ys_torch = ode_solver(y0, mu_t, args.dT, args.n_T, _dxdt, args.gradient_zero_from).detach().numpy()

    # Determine an error tolerance
    mean_diff = np.mean(np.abs(ys_torch[-1] - ode_ground_truth))
    median_diff = np.median(np.abs(ys_torch[-1] - ode_ground_truth))
    assert mean_diff < ode_test_case.ODE_ERROR_TOLERANCE
    assert median_diff < ode_test_case.ODE_ERROR_TOLERANCE


if __name__ == '__main__':

    pytest.main(args=['-sv', os.path.abspath(__file__)])
