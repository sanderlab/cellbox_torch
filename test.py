import pytest
import os
import glob
import torch
import tensorflow as tf
import numpy as np
import cellbox

from cellbox.kernel_torch import get_ode_solver, get_envelope, get_dxdt
from cellbox.utils_torch import loss

from test_utils.test_cases import KernelConfig, \
    ODETestCase, ODE_PARAMETRIZED_TESTS, DATALOADER_PARAMETRIZED_TESTS, \
    LOSS_PARAMETRIZED_TESTS

#from test_utils.dataloader import get_dataloader, yield_data_from_tensorflow_dataloader, yield_data_from_pytorch_dataloader, \
#    s2c_row_inds, loo_row_inds


#def test_model():
#    os.system('python scripts/main.py -config=configs/Example.minimal.json')
#    files = glob.glob('results/Debugging_*/seed_000/3_best.W*')
#    assert False

#################################################### Tests for dataloaders ####################################################

# Test for correct indices included in the data
@pytest.mark.parametrize("dataloader_test_case", DATALOADER_PARAMETRIZED_TESTS)
def test_dataloader_pos(dataloader_test_case):
    """
    Test Pytorch's dataset based on Tensorflow's ground truth
    """
    seed = dataloader_test_case.seed
    cfg = dataloader_test_case.cfg
    np.random.seed(seed)
    cfg = cellbox.dataset_torch.factory(cfg)
    try:
        os.remove("random_pos.csv")
    except:
        pass
    dataset = cfg.dataset
    assert np.all(dataset["train_pos"] == dataloader_test_case.ground_truth["train_pos"])
    assert np.all(dataset["valid_pos"] == dataloader_test_case.ground_truth["valid_pos"])
    assert np.all(dataset["test_pos"] == dataloader_test_case.ground_truth["test_pos"])


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



#################################################### Tests for util functions ####################################################

# Test for loss function
@pytest.mark.parametrize("loss_test_case", LOSS_PARAMETRIZED_TESTS)
def test_loss_fn(loss_test_case):
    """
    Test Pytorch's loss based on Tensorflow's ground truth
    """
    x_gold, x_hat, W = \
        torch.tensor(loss_test_case.x_gold, dtype=torch.float32), \
        torch.tensor(loss_test_case.x_hat, dtype=torch.float32), \
        torch.tensor(loss_test_case.W, dtype=torch.float32)
    l1, l2, mode = \
        torch.tensor(loss_test_case.l1, dtype=torch.float32), \
        torch.tensor(loss_test_case.l2, dtype=torch.float32), \
        loss_test_case.mode
    total_loss_tor = loss_test_case.total_loss_tor
    loss_mse_tor = loss_test_case.loss_mse_tor

    if mode == "expr":
        total_loss, loss_mse = loss(x_gold, x_hat, W, l1, l2, weight=x_gold)
    else:
        total_loss, loss_mse = loss(x_gold, x_hat, W, l1, l2)

    assert np.abs(total_loss.item() - loss_test_case.total_loss) < total_loss_tor
    assert np.abs(loss_mse.item() - loss_test_case.loss_mse) < loss_mse_tor



#################################################### Tests for model training ####################################################

# Test for the first feedforward pass of the model


# Test for the predictions and loss values after a long training


if __name__ == '__main__':

    pytest.main(args=['-sv', os.path.abspath(__file__)])
