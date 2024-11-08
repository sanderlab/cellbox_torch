import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
import time
import glob
from cellbox.utils_torch import optimize, TimeLogger

def _forward_pass(model, x, y, args):
    """
    Perform a forward pass for the model, then also compute the loss function
    Args:
        - model: The perturbation model
        - x: The torch tensor as model input
        - y: The ground truth output
        - args: The Config object
    Returns:
        - convergence_metric: A torch tensor containing the mean, std, and
          gradient of the last steps of the ODE solver
        - yhat: The model prediction given x as input
        - loss_total: The total loss, including the MSE loss between y and yhat,
          and the L1 and L2 loss on the model parameters
        - loss_mse: The MSE loss between y and yhat
    """
    if args.pert_form == "by u":
        prediction = model(torch.zeros((args.n_x, 1), dtype=torch.float32).to(args.device), x.to(args.device))
    elif args.pert_form == "fix x":
        prediction = model(x.T.to(args.device), x.to(args.device))
    convergence_metric, yhat = prediction

    for param in model.named_parameters():
        if param[0] == "params.W":
            param_mat = param[1]
            break
        else:
            param_mat = None

    if args.weight_loss == "expr":
        loss_total, loss_mse = args.loss_fn(y.to(args.device), yhat, param_mat, l1=args.l1_lambda, l2=args.l2_lambda, weight=y.to(args.device))
    else:
        loss_total, loss_mse = args.loss_fn(y.to(args.device), yhat, param_mat, l1=args.l1_lambda, l2=args.l2_lambda)

    return convergence_metric, yhat, loss_total, loss_mse


def train_substage(model, lr_val, l1_lambda, l2_lambda, n_epoch, n_iter, n_iter_buffer, n_iter_patience, args):
    """
    Training function that does one stage of training. The stage training can be repeated and modified to give better
    training result.

    Args:
        model (CellBox): an CellBox instance
        lr_val (float): learning rate (read in from config file)
        l1_lambda (float): l1 regularization weight
        l2_lambda (float): l2 regularization weight
        n_epoch (int): maximum number of epochs
        n_iter (int): maximum number of iterations
        n_iter_buffer (int): training loss moving average window
        n_iter_patience (int): training loss tolerance
        args: Args or configs
    """

    stages = glob.glob("*best*.csv")
    try:
        substage_i = 1 + max([int(stage[0]) for stage in stages])
    except Exception:
        substage_i = 1

    best_params = Screenshot(args, n_iter_buffer)

    n_unchanged = 0
    idx_iter = 0
    args.logger.log("--------- lr: {}\tl1: {}\tl2: {}\t".format(lr_val, l1_lambda, l2_lambda))
    args.l1_lambda = l1_lambda
    args.l2_lambda = l2_lambda
    args.lr = lr_val
    args.optimizer = optimize(
        model.parameters(),
        lr=args.lr
    )

    for idx_epoch in range(n_epoch):

        if idx_iter > n_iter or n_unchanged > n_iter_patience:
            if idx_iter > n_iter: 
                args.logger.log("Ended substage due to iteration exceeding maximum value")
            if n_unchanged > n_iter_patience:
                args.logger.log("Ended substage due to loss not improving exceeding maximum number of iterations")
            break

        for i, train_minibatch in enumerate(args.iter_train):
            # Each train_minibatch has shape of (batch_size, num_features)
            x_train, y_train = train_minibatch

            if idx_iter > n_iter or n_unchanged > n_iter_patience:
                break

            # Do one forward pass
            t0 = time.perf_counter()
            model.train()
            args.optimizer.zero_grad()
            convergence_metric, yhat, loss_train_i, loss_train_mse_i = _forward_pass(model, x_train, y_train, args)
            loss_train_i.backward()
            args.optimizer.step()

            # Record training
            with torch.no_grad():
                model.eval()
                valid_minibatch = iter(args.iter_monitor)
                x_valid, y_valid = next(valid_minibatch)
                convergence_metric, yhat, loss_valid_i, loss_valid_mse_i = _forward_pass(model, x_valid, y_valid, args)

            # Record results to screenshot
            new_loss = best_params.avg_n_iters_loss(loss_valid_i.item())
            if args.export_verbose > 0:
                print(
                    f"Substage:{substage_i}\tEpoch:{idx_epoch}/{n_epoch}\tIteration: {idx_iter}/{n_iter}" +
                    f"\tloss (train):{loss_train_i:1.6f}\tloss (buffer on valid):{new_loss:1.6f}" +
                    f"\tbest:{best_params.loss_min:1.6f}\tTolerance: {n_unchanged}/{n_iter_patience}"
                    )
            
            append_record("record_eval.csv",
                          [idx_epoch, idx_iter, loss_train_i.item(), loss_valid_i.item(), loss_train_mse_i.item(),
                           loss_valid_mse_i.item(), None, time.perf_counter() - t0])

            # Early stopping
            idx_iter += 1
            if new_loss < best_params.loss_min:
                n_unchanged = 0
                best_params.screenshot(model, substage_i, args=args,
                                       node_index=args.dataset['node_index'], loss_min=new_loss)
            else:
                n_unchanged += 1

        for k, v in best_params.items():
            assert type(v) == pd.DataFrame, print(k)

    # Evaluation on valid set
    t0 = time.perf_counter()
    loss_valid_i = eval_model(
        args, args.iter_monitor, model, return_value="loss_full", n_batches_eval=args.n_batches_eval
    )
    loss_valid_mse_i = eval_model(
        args, args.iter_monitor, model, return_value="loss_mse", n_batches_eval=args.n_batches_eval
    )
    append_record("record_eval.csv", [-1, None, None, loss_valid_i, None, loss_valid_mse_i, None, time.perf_counter() - t0])

    # Evaluation on test set
    t0 = time.perf_counter()
    loss_test_mse = eval_model(
        args, args.iter_eval, model, return_value="loss_mse", n_batches_eval=args.n_batches_eval
    )
    append_record("record_eval.csv", [-1, None, None, None, None, None, loss_test_mse, time.perf_counter() - t0])

    # Save results
    best_params.save()
    args.logger.log("------------------ Substage {} finished!-------------------".format(substage_i))
    save_model(model, f"./{args.ckpt_name}")

    return best_params


def append_record(filename, contents):
    """define function for appending training record"""
    with open(filename, 'a') as f:
        for content in contents:
            f.write('{},'.format(content))
        f.write('\n')


def eval_model(args, eval_iter, model, return_value, return_avg=True, n_batches_eval=None):
    """ 
    Simulate the model for evaluation.
    Args:
        - args (Config): The Config object
        - eval_iter (DataLoader): The dataloader used for evaluation
        - model (nn.Module): The perturbation model
        - return_value (str): The specific variable to return
        - return_avg (bool): Whether to return an average value
        - n_batches_eval (int): The number of batches for early stopping
    Returns:
        - out (np.array): An array of the variable of interest
    """
    with torch.no_grad():
        model.eval()
        counter = 0
        eval_results = []
        for item in eval_iter:
            pert, expr = item
            convergence_metric, yhat, loss_full, loss_mse = _forward_pass(model, pert, expr, args)

            if return_value == "prediction":
                eval_results.append(yhat.detach().cpu().numpy())
            elif return_value == "loss_full":
                eval_results.append(loss_full.detach().cpu().numpy())
            elif return_value == "loss_mse":
                eval_results.append(loss_mse.detach().cpu().numpy())
            
            counter += 1
            if n_batches_eval is not None and counter > n_batches_eval:
                break
        print(eval_results)
        if return_avg:
            return np.mean(np.array(eval_results), axis=0)
        return np.vstack(eval_results)
    

def save_model(model, save_dir):
    """ Save the model """
    torch.save(model.state_dict(), save_dir)


def train_model(model, args):
    """
    Train the model specified by the substages in args. This function loops through the
    substages and calls train_substage for each substage.
    Args:
        - model (nn.Module): The perturbation model
        - args (Config): The Config model
    """
    args.logger = TimeLogger(time_logger_step=1, hierachy=2)
    model = model[0].to(args.device)

    try:
        print(f'Load existing model at ./{args.ckpt_name}...')
        model.load_state_dict(torch.load(f"./{args.ckpt_name}"))
    except Exception:
        print(f'Create new model at ./{args.ckpt_name}...')

    # Training
    for substage in args.sub_stages:
        n_iter_buffer = substage['n_iter_buffer'] if 'n_iter_buffer' in substage else args.n_iter_buffer
        n_iter = substage['n_iter'] if 'n_iter' in substage else args.n_iter
        n_iter_patience = substage['n_iter_patience'] if 'n_iter_patience' in substage else args.n_iter_patience
        n_epoch = substage['n_epoch'] if 'n_epoch' in substage else args.n_epoch
        l1 = substage['l1lambda'] if 'l1lambda' in substage else args.l1lambda if hasattr(args, 'l1lambda') else 0
        l2 = substage['l2lambda'] if 'l2lambda' in substage else args.l2lambda if hasattr(args, 'l2lambda') else 0
        train_substage(model, substage['lr_val'], l1_lambda=l1, l2_lambda=l2, n_epoch=n_epoch,
                       n_iter=n_iter, n_iter_buffer=n_iter_buffer, n_iter_patience=n_iter_patience, args=args)
        

class Screenshot(dict):
    """summarize the model"""
    def __init__(self, args, n_iter_buffer):
        # initialize loss_min
        super().__init__()
        self.loss_min = 1000
        # initialize tuning_metric
        self.saved_losses = [self.loss_min]
        self.n_iter_buffer = n_iter_buffer
        # initialize verbose
        self.summary = {}
        self.summary = {}
        self.substage_i = []
        self.export_verbose = args.export_verbose
        self.args = args

    def avg_n_iters_loss(self, new_loss):
        """average the last few losses"""
        self.saved_losses = self.saved_losses + [new_loss]
        self.saved_losses = self.saved_losses[-self.n_iter_buffer:]
        return sum(self.saved_losses) / len(self.saved_losses)

    def screenshot(self, model, substage_i, node_index, loss_min, args):
        """evaluate models"""
        self.substage_i = substage_i
        self.loss_min = loss_min

        # Save the variable weights associated with each of the conditions in a csv file
        if self.export_verbose > 0:
            #layer = model.W
            params = model.state_dict()
            new_params = {}
            for item in params:
                try:
                    new_params[item] = pd.DataFrame(params[item].detach().numpy(), index=node_index[0])
                except Exception:
                    new_params[item] = pd.DataFrame(params[item].detach().numpy())
            self.update(new_params)

        if self.export_verbose > 1 or self.export_verbose == -1:  # no params but y_hat
            y_hat = eval_model(args, args.iter_eval, model, return_value="prediction", return_avg=False)
            y_hat = pd.DataFrame(y_hat, columns=node_index[0])
            self.update({'y_hat': y_hat})

        if self.export_verbose > 2:
            try:
                with torch.no_grad():
                    model.eval()
                    # Run summary on train set
                    converge_train_mat, converge_eval_mat, converge_test_mat = [], [], []
                    for item in args.iter_train:
                        pert, expr = item
                        convergence_metric_train, _, _, _ = _forward_pass(model, pert, expr, args)
                        converge_train_mat.append(convergence_metric_train.detach().numpy())

                    # Run summary on eval set
                    for item in args.iter_monitor:
                        pert, expr = item
                        convergence_metric_eval, _, _, _ = _forward_pass(model, pert, expr, args)
                        converge_eval_mat.append(convergence_metric_eval.detach().numpy())
                    
                    # Run summary on test set
                    for item in args.iter_eval:
                        pert, expr = item
                        convergence_metric_test, _, _, _ = _forward_pass(model, pert, expr, args)
                        converge_test_mat.append(convergence_metric_test.detach().numpy())

                # Concatenate the results:
                converge_train_mat = np.concatenate(converge_train_mat, axis=1)
                converge_test_mat = np.concatenate(converge_test_mat, axis=1)
                converge_eval_mat = np.concatenate(converge_eval_mat, axis=1)
                
                # Summarize performance
                cols = [node_index.values + '_mean', node_index.values + '_sd', node_index.values + '_dxdt']
                cols = np.squeeze(np.concatenate(cols)).tolist()
                summary_train = pd.DataFrame(converge_train_mat.T, columns=cols)
                summary_test = pd.DataFrame(converge_test_mat.T, columns=cols)
                summary_valid = pd.DataFrame(converge_eval_mat.T, columns=cols)
                self.update(
                    {'summary_train': summary_train, 'summary_test': summary_test, 'summary_valid': summary_valid}
                )

            except Exception as e:
                print(e)

    def save(self):
        """save model parameters"""
        for file in glob.glob(str(self.substage_i) + "_best.*.csv"):
            os.remove(file)
        for key in self:
            self[key].to_csv("{}_best.{}.loss.{}.csv".format(self.substage_i, key, self.loss_min))
        
