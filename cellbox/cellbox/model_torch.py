import numpy as np
import os
import pandas as pd
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
            weights_path = self.args.weights
            ext = os.path.splitext(weights_path)[1].lower()
            if ext in [".npy", ".npz"]:
                loaded = np.load(weights_path, allow_pickle=False)
                if isinstance(loaded, np.lib.npyio.NpzFile):
                    # Use the first array if multiple are present
                    keys = list(loaded.keys())
                    if len(keys) == 0:
                        raise ValueError(f"Weights file '{weights_path}' contains no arrays")
                    W_np = loaded[keys[0]]
                else:
                    W_np = loaded
            elif ext in [".csv", ".tsv", ".txt"]:
                sep = "," if ext == ".csv" else None
                W_np = pd.read_csv(weights_path, header=None, sep=sep).values
            else:
                raise ValueError(f"Unsupported weights file extension '{ext}'. Use .npy/.npz or .csv")

            if W_np.shape != (n_x, n_x):
                raise ValueError(
                    f"Weights matrix shape {W_np.shape} does not match expected (n_x, n_x)=({n_x}, {n_x}). "
                    f"Check that the file contains a full {n_x}x{n_x} adjacency in the correct node order."
                )

            W = torch.tensor(W_np, dtype=torch.float32)
        else:
            W = torch.normal(mean=0.01, std=1.0, size=(n_x, n_x), dtype=torch.float32)

        # Build mask for parameter initialization, optionally combining with a user-provided mask
        extra_mask_init = None
        if hasattr(self.args, 'extra_mask') and getattr(self.args, 'extra_mask') is not None:
            extra_mask_init = getattr(self.args, 'extra_mask')
        elif hasattr(self.args, 'extra_mask_path') and getattr(self.args, 'extra_mask_path') not in [None, "None", ""]:
            mask_path = getattr(self.args, 'extra_mask_path')
            ext = os.path.splitext(mask_path)[1].lower()
            if ext in [".npy", ".npz"]:
                loaded_m = np.load(mask_path, allow_pickle=False)
                if isinstance(loaded_m, np.lib.npyio.NpzFile):
                    keys = list(loaded_m.keys())
                    if len(keys) == 0:
                        raise ValueError(f"Mask file '{mask_path}' contains no arrays")
                    extra_mask_init = loaded_m[keys[0]]
                else:
                    extra_mask_init = loaded_m
            elif ext in [".csv", ".tsv", ".txt"]:
                sep = "," if ext == ".csv" else None
                extra_mask_init = pd.read_csv(mask_path, header=None, sep=sep).values
            else:
                raise ValueError(f"Unsupported mask file extension '{ext}'. Use .npy/.npz or .csv")

        W_mask = self._get_mask(extra_mask_init)
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

    def _get_mask(self, extra_mask=None):
        """
        Get the adjacency mask. Optionally combine with an external mask such that a connection
        is enabled only if allowed by BOTH masks (logical AND via elementwise multiply).

        Args:
            - extra_mask: optional mask to combine with the base mask. Can be a torch.Tensor or
                          numpy.ndarray of shape (n_x, n_x). Non-zero values are treated as 1.
        """
        W_mask_np = (1.0 - np.diag(np.ones([self.n_x])))
        W_mask_np[self.args.n_activity_nodes:, :] = np.zeros([self.n_x - self.args.n_activity_nodes, self.n_x])
        W_mask_np[:, self.args.n_protein_nodes:self.args.n_activity_nodes] = np.zeros([self.n_x, self.args.n_activity_nodes - self.args.n_protein_nodes])
        W_mask_np[self.args.n_protein_nodes:self.args.n_activity_nodes, self.args.n_activity_nodes:] = np.zeros([self.args.n_activity_nodes - self.args.n_protein_nodes,
                                                                                self.n_x - self.args.n_activity_nodes])

        W_mask = torch.tensor(W_mask_np, dtype=torch.float32)

        if extra_mask is not None:
            if isinstance(extra_mask, np.ndarray):
                extra = torch.tensor((extra_mask != 0).astype(np.float32), dtype=torch.float32)
            elif torch.is_tensor(extra_mask):
                extra = extra_mask.detach().to(dtype=torch.float32)
                extra = (extra != 0).to(dtype=torch.float32)
            else:
                raise TypeError("extra_mask must be a numpy.ndarray or torch.Tensor")
            if extra.shape != (self.n_x, self.n_x):
                raise ValueError(f"extra_mask shape {tuple(extra.shape)} must match (n_x, n_x)=({self.n_x}, {self.n_x})")
            W_mask = W_mask * extra

        return W_mask

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
