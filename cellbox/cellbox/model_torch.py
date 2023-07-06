import numpy as np
import torch
import torch.nn as nn
import cellbox.kernel_torch

from cellbox.utils_torch import loss, optimize


def factory(args):
    """define model type based on configuration input"""
    #if args.model == 'CellBox':
    #    return CellBox(args).build()
    # Deprecated for now, use scikit-learn instead
    # TODO: update the co-expression models
    # if args.model == 'CoExp':
    #     return CoExp(args).build()
    # if args.model == 'CoExp_nonlinear':
    #     return CoExpNonlinear(args).build()
    if args.model == 'CellBox':
        model = CellBox(args)
        args = get_ops(args, model)
        return model, args
    if args.model == 'LinReg':
        model = LinReg(args)
        args = get_ops(args, model)
        return model, args
    #if args.model == 'NN':
    #    return NN(args).build()
    # TODO: baysian model
    # if args.model == 'Bayesian':
    #     return BN(args).build()


class PertBio(nn.Module):
    """define abstract perturbation model"""
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.n_x = args.n_x
        #self.pert_in, self.expr_out = args.pert_in, args.expr_out
        self.iter_train, self.iter_monitor, self.iter_eval = args.iter_train, args.iter_monitor, args.iter_eval
        #self.train_x, self.train_y = self.iter_train.get_next()
        #self.monitor_x, self.monitor_y = self.iter_monitor.get_next()
        #self.eval_x, self.eval_y = self.iter_eval.get_next()
        self.l1_lambda, self.l2_lambda = self.args.l1_lambda_placeholder, self.args.l2_lambda_placeholder
        self.lr = self.args.lr
        self.params = {}
        self.build()

    #def get_ops(self):
    #    """get operators for tensorflow"""
    #    # Do we need this at all for Pytorch?
    #    pass
        #if self.args.weight_loss == 'expr':
        #    self.train_loss, self.train_mse_loss = loss(self.train_y, self.train_yhat, self.params['W'],
        #                                                self.l1_lambda, self.l2_lambda, weight=self.train_y)
        #    self.monitor_loss, self.monitor_mse_loss = loss(self.monitor_y, self.monitor_yhat, self.params['W'],
        #                                                    self.l1_lambda, self.l2_lambda, weight=self.monitor_y)
        #    self.eval_loss, self.eval_mse_loss = loss(self.eval_y, self.eval_yhat, self.params['W'],
        #                                              self.l1_lambda, self.l2_lambda, weight=self.eval_y)
        #elif self.args.weight_loss == 'None':
        #    self.train_loss, self.train_mse_loss = loss(self.train_y, self.train_yhat, self.params['W'],
        #                                                self.l1_lambda, self.l2_lambda)
        #    self.monitor_loss, self.monitor_mse_loss = loss(self.monitor_y, self.monitor_yhat, self.params['W'],
        #                                                    self.l1_lambda, self.l2_lambda)
        #    self.eval_loss, self.eval_mse_loss = loss(self.eval_y, self.eval_yhat, self.params['W'],
        #                                              self.l1_lambda, self.l2_lambda)
        #self.op_optimize = optimize(self.train_loss, self.lr)

    def get_variables(self):
        """get model parameters (overwritten by model configuration)"""
        raise NotImplementedError

    def build(self):
        """get model parameters (overwritten by model configuration)"""
        raise NotImplementedError
    
    def forward(self, x, mu):
        """forward propagation (overwritten by model configuration)"""
        raise NotImplementedError

    #def build(self):
    #    """build model"""
    #    # Do we need this at all for Pytorch?
    #    self.params = {}
    #    self.get_variables()
    #    self.train_yhat = self.forward(self.train_y0, self.train_x)
    #    self.monitor_yhat = self.forward(self.monitor_y0, self.monitor_x)
    #    self.eval_yhat = self.forward(self.eval_y0, self.train_x)
    #    self.get_ops()
    #    return self


class CellBox(PertBio):
    def build(self):
        """
        Get the nn Parameters
        """
        n_x, n_protein_nodes, n_activity_nodes = self.n_x, self.args.n_protein_nodes, self.args.n_activity_nodes
        self.params = nn.ParameterDict()

        W = nn.Parameter(torch.normal(mean=0.01, std=1.0, size=(n_x, n_x)))
        W_mask = (1.0 - np.diag(np.ones([n_x])))
        W_mask[n_activity_nodes:, :] = np.zeros([n_x - n_activity_nodes, n_x])
        W_mask[:, n_protein_nodes:n_activity_nodes] = np.zeros([n_x, n_activity_nodes - n_protein_nodes])
        W_mask[n_protein_nodes:n_activity_nodes, n_activity_nodes:] = np.zeros([n_activity_nodes - n_protein_nodes,
                                                                                n_x - n_activity_nodes])
        final_W = (torch.from_numpy(W_mask)*W).type(torch.float32)
        self.params['W'] = nn.Parameter(final_W)

        eps = nn.Parameter(torch.ones((n_x, 1)))
        alpha = nn.Parameter(torch.ones((n_x, 1)))
        self.params['alpha'] = nn.functional.softplus(alpha)
        self.params['eps'] = nn.functional.softplus(eps)

        if self.args.envelope == 2:
            psi = nn.Parameter(torch.ones((n_x, 1)))
            self.params['psi'] = torch.nn.functional.softplus(psi)

        if self.args.pert_form == 'by u':
            #y0 = tf.constant(np.zeros((self.n_x, 1)), name="x_init", dtype=tf.float32)
            #self.train_y0 = y0
            #self.monitor_y0 = y0
            #self.eval_y0 = y0
            self.gradient_zero_from = None
        elif self.args.pert_form == 'fix x':  # fix level of node x (here y) by input perturbation u (here x)
            #self.train_y0 = tf.transpose(self.train_x)
            #self.monitor_y0 = tf.transpose(self.monitor_x)
            #self.eval_y0 = tf.transpose(self.eval_x)
            self.gradient_zero_from = self.args.n_activity_nodes

        self.envelope_fn = cellbox.kernel_torch.get_envelope(self.args)
        self.ode_solver = cellbox.kernel_torch.get_ode_solver(self.args)
        self._dxdt = cellbox.kernel_torch.get_dxdt(self.args, self.params)

    def get_variables(self):
        """
        Initialize parameters in the Hopfield equation

        Mutates:
            self.params(dict):{
                W (tf.Variable): interaction matrix with constraints enforced, , shape: [n_x, n_x]
                alpha (tf.Variable): alpha, shape: [n_x, 1]
                eps (tf.Variable): eps, shape: [n_x, 1]
            }
        """
        
        """
            Enforce constraints  (i: recipient)
            no self regulation wii=0
            ingoing wij for drug nodes (88th to 99th) = 0 [n_activity_nodes 87: ]
                            w [87:99,_] = 0
            outgoing wij for phenotypic nodes (83th to 87th) [n_protein_nodes 82 : n_activity_nodes 87]
                            w [_, 82:87] = 0
            ingoing wij for phenotypic nodes from drug ndoes (direct) [n_protein_nodes 82 : n_activity_nodes 87]
                            w [82:87, 87:99] = 0
        """

    def forward(self, y0, mu):
        mu_t = torch.transpose(mu, 0, 1)
        ys = self.ode_solver(y0, mu_t, self.args.dT, self.args.n_T, self._dxdt, self.gradient_zero_from)
        # [n_T, n_x, batch_size]
        ys = ys[-self.args.ode_last_steps:]
        # [n_iter_tail, n_x, batch_size]
        mean = torch.mean(ys, dim=0)
        sd = torch.std(ys, dim=0)
        yhat = torch.transpose(ys[-1], 0, 1)
        dxdt = self._dxdt(ys[-1], mu_t)
        # [n_x, batch_size] for last ODE step
        convergence_metric = torch.cat([mean, sd, dxdt], dim=0)
        return convergence_metric, yhat
    

class LinReg(PertBio):
    """linear regression model"""
    def get_variables(self):
        self.W = nn.Linear(
            in_features=self.n_x,
            out_features=self.n_x,
            bias=True
        )

    def forward(self, x, mu):
        return self.W(mu)
    

def get_ops(args, model):
    args.loss_fn = loss
    args.optimizer = optimize(
        model.parameters(),
        lr=args.lr
    )
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return args
