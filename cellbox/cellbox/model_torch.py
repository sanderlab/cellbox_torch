import numpy as np
import torch
import torch.nn as nn
import cellbox.kernel_torch

from cellbox.utils_torch import loss, optimize


def factory(args):
    """
    Define model type based on configuration input. Currently supporting only 'CellBox'
    Args:
        - args: the Config object
    Returns:
        - model: the Pytorch model
        - args: the updated Config object
    """
    if args.model == 'CellBox':
        model = CellBox(args)
        args = get_ops(args, model)
        return model, args
    if args.model == 'LinReg':
        model = LinReg(args)
        args = get_ops(args, model)
        return model, args


class PertBio(nn.Module):
    """
    Define abstract perturbation model. All subsequent models are inherited from this model.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.n_x = args.n_x
        self.iter_train, self.iter_monitor, self.iter_eval = args.iter_train, args.iter_monitor, args.iter_eval
        self.l1_lambda, self.l2_lambda = 0.0, 0.0
        self.lr = self.args.lr
        self.params = {}
        self.build()

    def get_variables(self):
        """get model parameters (overwritten by model configuration)"""
        raise NotImplementedError

    def build(self):
        """get model parameters (overwritten by model configuration)"""
        raise NotImplementedError
    
    def forward(self, x, mu):
        """forward propagation (overwritten by model configuration)"""
        raise NotImplementedError


class CellBox(PertBio):
    def build(self):
        """
        Initialize the CellBox model
        """
        n_x, n_protein_nodes, n_activity_nodes = self.n_x, self.args.n_protein_nodes, self.args.n_activity_nodes
        self.params = nn.ParameterDict()

        if self.args.weights and self.args.weights != "None":
            with open(self.args.weights, "rb") as f:
                W = torch.tensor(np.load(f), dtype=torch.float32)
        else:
            W = torch.normal(mean=0.01, std=1.0, size=(n_x, n_x), dtype=torch.float32)

        W_mask = self._get_mask()
        self.params['W'] = nn.Parameter(W_mask*W, requires_grad=True)
        eps = nn.Parameter(torch.ones((n_x, 1), dtype=torch.float32), requires_grad=True)
        alpha = nn.Parameter(torch.ones((n_x, 1), dtype=torch.float32), requires_grad=True)
        self.params['alpha'] = nn.functional.softplus(alpha)
        self.params['eps'] = nn.functional.softplus(eps)

        if self.args.envelope == 2:
            psi = nn.Parameter(torch.ones((n_x, 1), dtype=torch.float32), requires_grad=True)
            self.params['psi'] = torch.nn.functional.softplus(psi)

        if self.args.pert_form == 'by u':
            self.gradient_zero_from = None
        elif self.args.pert_form == 'fix x':  # fix level of node x (here y) by input perturbation u (here x)
            self.gradient_zero_from = self.args.n_activity_nodes

        self.envelope_fn = cellbox.kernel_torch.get_envelope(self.args)
        self.ode_solver = cellbox.kernel_torch.get_ode_solver(self.args)
        self._dxdt = cellbox.kernel_torch.get_dxdt(self.args, self.params)

    def _get_mask(self):
        """
        Get the mask of the tensors. The mask is applied during the forward pass to disable
        certain connections in the adjacency matrix.
        """
        W_mask = (1.0 - np.diag(np.ones([self.n_x])))
        W_mask[self.args.n_activity_nodes:, :] = np.zeros([self.n_x - self.args.n_activity_nodes, self.n_x])
        W_mask[:, self.args.n_protein_nodes:self.args.n_activity_nodes] = np.zeros([self.n_x, self.args.n_activity_nodes - self.args.n_protein_nodes])
        W_mask[self.args.n_protein_nodes:self.args.n_activity_nodes, self.args.n_activity_nodes:] = np.zeros([self.args.n_activity_nodes - self.args.n_protein_nodes,
                                                                                self.n_x - self.args.n_activity_nodes])
        #final_W = (torch.from_numpy(W_mask)*W).type(torch.float32)
        #self.params['W'] = self.params["W"] * (torch.tensor(W_mask, dtype=torch.float32))
        return torch.tensor(W_mask, dtype=torch.float32)

    def forward(self, y0, mu):
        mu_t = torch.transpose(mu, 0, 1)
        mask = self._get_mask()
        ys = self.ode_solver(y0, mu_t, self.args.dT, self.args.n_T, self._dxdt, self.gradient_zero_from, mask=mask)
        # [n_T, n_x, batch_size]
        ys = ys[-self.args.ode_last_steps:]
        # [n_iter_tail, n_x, batch_size]
        #self.mask()
        mean = torch.mean(ys, dim=0)
        sd = torch.std(ys, dim=0)
        yhat = torch.transpose(ys[-1], 0, 1)
        dxdt = self._dxdt(ys[-1], mu_t)
        # [n_x, batch_size] for last ODE step
        convergence_metric = torch.cat([mean, sd, dxdt], dim=0)
        return convergence_metric, yhat
    

class LinReg(PertBio):
    """linear regression model"""
    def build(self):
        self.W = nn.Linear(
            in_features=self.n_x,
            out_features=self.n_x,
            bias=True
        )

    def forward(self, x, mu):
        ys = self.W(mu)
        mean = torch.mean(ys, dim=0)
        sd = torch.std(ys, dim=0)
        convergence_metric = torch.cat([mean, sd], dim=0)
        return convergence_metric, ys
    

def get_ops(args, model):
    """
    Initialize the loss function, optimizer, and device for training the perturbation model
    """
    args.loss_fn = loss
    args.optimizer = optimize(
        model.parameters(),
        lr=args.lr
    )
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return args
