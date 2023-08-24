import pytest
import os
import glob
import torch
import numpy as np
import cellbox

from cellbox.kernel_torch import get_ode_solver, get_envelope, get_dxdt
from cellbox.utils_torch import loss

from test_utils.test_cases import KernelConfig, \
    ODETestCase, ODE_PARAMETRIZED_TESTS, DATALOADER_PARAMETRIZED_TESTS, \
    LOSS_PARAMETRIZED_TESTS, FEEDFORWARD_PARAMETRIZED_TESTS

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
@pytest.mark.parametrize("feedforward_test_case", FEEDFORWARD_PARAMETRIZED_TESTS)
def test_feedforward(feedforward_test_case):
    """
    Test Pytorch's first feedforward
    """
    data_in = feedforward_test_case.data_in
    data_out = feedforward_test_case.data_out
    torch_args = feedforward_test_case.args
    torch_cellbox = cellbox.model_torch.factory(torch_args)[0]
    for w in torch_cellbox.named_parameters():
        if w[0] == "params.W": w[1].data = torch.tensor(data_in["W"], dtype=torch.float32)
    l1_lambda = data_in["l1_lambda"]
    l2_lambda = data_in["l2_lambda"]
    loss_fn = cellbox.utils_torch.loss

    torch_cellbox.train()
    if torch_args.pert_form == "by u":
        prediction = torch_cellbox(
            torch.zeros((torch_args.n_x, 1), dtype=torch.float32), 
            torch.tensor(data_in["inp"], dtype=torch.float32)
        )
    elif torch_args.pert_form == "fix x":
        prediction = torch_cellbox(
            torch.tensor(data_in["inp"].T, dtype=torch.float32),
            torch.tensor(data_in["inp"], dtype=torch.float32)
        )
    convergence_metric, yhat = prediction

    for param in torch_cellbox.named_parameters():
        if param[0] == "params.W":
            param_mat = param[1]
            break
    loss_train_i_torch, loss_train_mse_i_torch = loss_fn(
        torch.tensor(data_in["out"], dtype=torch.float32), 
        yhat,
        param_mat, 
        l1=l1_lambda, 
        l2=l2_lambda)

    # Test if the loss after one feedforward is similar
    assert abs(loss_train_i_torch.item() - data_out["loss_train"]) <= feedforward_test_case.tolerance["loss_total"]
    assert abs(loss_train_mse_i_torch.item() - data_out["loss_train_mse"]) <= feedforward_test_case.tolerance["loss_mse"]
    # Test if the predicted yhat is similar
    assert \
        np.all(np.abs(data_out['yhat'] - yhat.detach().cpu().numpy()) < feedforward_test_case.tolerance["yhat"])
    # Test if the mask on yhat is similar


# Test for the predictions and loss values after a long training



if __name__ == '__main__':

    pytest.main(args=['-sv', os.path.abspath(__file__)])
