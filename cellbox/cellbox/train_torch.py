import numpy as np
import torch
import torch.nn as nn
import time
import glob
from cellbox.utils import TimeLogger

def train_substage(model, lr_val, l1_lambda, l2_lambda, n_epoch, n_iter, n_iter_buffer, n_iter_patience, args):
    """
    Training function that does one stage of training. The stage training can be repeated and modified to give better
    training result.

    Args:
        model (CellBox): an CellBox instance
        sess (tf.Session): current session, need reinitialization for every nT
        lr_val (float): learning rate (read in from config file)
        l1_lambda (float): l1 regularization weight
        l2_lambda (float): l2 regularization weight
        n_epoch (int): maximum number of epochs
        n_iter (int): maximum number of iterations
        n_iter_buffer (int): training loss moving average window
        n_iter_patience (int): training loss tolerance
        args: Args or configs
    """

    # Let's just assume that args contains also the loss function, dataloaders, and the optimizer
    stages = glob.glob("*best*.csv")
    try:
        substage_i = 1 + max([int(stage[0]) for stage in stages])
    except Exception:
        substage_i = 1

    #best_params = Screenshot(args, n_iter_buffer)

    n_unchanged = 0
    idx_iter = 0
    #for key in args.feed_dicts:
    #    args.feed_dicts[key].update({
    #        model.lr: lr_val,
    #        model.l1_lambda: l1_lambda,
    #        model.l2_lambda: l2_lambda
    #    })
    args.logger.log("--------- lr: {}\tl1: {}\tl2: {}\t".format(lr_val, l1_lambda, l2_lambda))

    #sess.run(model.iter_monitor.initializer, feed_dict=args.feed_dicts['valid_set'])
    for idx_epoch in range(n_epoch):

        if idx_iter > n_iter or n_unchanged > n_iter_patience:
            break

        for i, train_minibatch in enumerate(args.iter_train):
            # Each train_minibatch has shape of (batch_size, num_features)

            if idx_iter > n_iter or n_unchanged > n_iter_patience:
                break

            # Do one forward pass
            t0 = time.perf_counter()
            args.optimizer.zero_grad()
            prediction = model(train_minibatch)
            


        #sess.run(model.iter_train.initializer, feed_dict=args.feed_dicts['train_set'])
        #while True:
        #    if idx_iter > n_iter or n_unchanged > n_iter_patience:
        #        break
        #    t0 = time.perf_counter()
        #    try:
        #        _, loss_train_i, loss_train_mse_i = sess.run(
        #            (model.op_optimize, model.train_loss, model.train_mse_loss), feed_dict=args.feed_dicts['train_set'])
        #    except OutOfRangeError:  # for iter_train
        #        break
#
        #    # record training
        #    loss_valid_i, loss_valid_mse_i = sess.run(
        #        (model.monitor_loss, model.monitor_mse_loss), feed_dict=args.feed_dicts['valid_set'])
        #    new_loss = best_params.avg_n_iters_loss(loss_valid_i)
        #    if args.export_verbose > 0:
        #        print(("Substage:{}\tEpoch:{}/{}\tIteration: {}/{}" + "\tloss (train):{:1.6f}\tloss (buffer on valid):"
        #               "{:1.6f}" + "\tbest:{:1.6f}\tTolerance: {}/{}").format(substage_i, idx_epoch, n_epoch, idx_iter,
        #                                                                      n_iter, loss_train_i, new_loss,
        #                                                                      best_params.loss_min, n_unchanged,
        #                                                                      n_iter_patience))
        #    append_record("record_eval.csv",
        #                  [idx_epoch, idx_iter, loss_train_i, loss_valid_i, loss_train_mse_i,
        #                   loss_valid_mse_i, None, time.perf_counter() - t0])
        #    # early stopping
        #    idx_iter += 1
        #    if new_loss < best_params.loss_min:
        #        n_unchanged = 0
        #        best_params.screenshot(sess, model, substage_i, args=args,
        #                               node_index=args.dataset['node_index'], loss_min=new_loss)
        #    else:
        #        n_unchanged += 1
#
    # Evaluation on valid set
    t0 = time.perf_counter()
    sess.run(model.iter_eval.initializer, feed_dict=args.feed_dicts['valid_set'])
    loss_valid_i, loss_valid_mse_i = eval_model(sess, model.iter_eval, (model.eval_loss, model.eval_mse_loss),
                                                args.feed_dicts['valid_set'], n_batches_eval=args.n_batches_eval)
    append_record("record_eval.csv", [-1, None, None, loss_valid_i, None, loss_valid_mse_i, None, time.perf_counter() - t0])

    # Evaluation on test set
    t0 = time.perf_counter()
    sess.run(model.iter_eval.initializer, feed_dict=args.feed_dicts['test_set'])
    loss_test_mse = eval_model(sess, model.iter_eval, model.eval_mse_loss,
                               args.feed_dicts['test_set'], n_batches_eval=args.n_batches_eval)
    append_record("record_eval.csv", [-1, None, None, None, None, None, loss_test_mse, time.perf_counter() - t0])

    best_params.save()
    args.logger.log("------------------ Substage {} finished!-------------------".format(substage_i))
    save_model(args.saver, sess, './' + args.ckpt_name)


def train_model(model, args):
    """Train the model"""
    args.logger = TimeLogger(time_logger_step=1, hierachy=2)

    # Check if all variables in scope
    # TODO: put variables under appropriate scopes
    #try:
    #    args.saver.restore(sess, './' + args.ckpt_name)
    #    print('Load existing model at {}...'.format(args.ckpt_name))
    #except Exception:
    #    print('Create new model at {}...'.format(args.ckpt_name))

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
