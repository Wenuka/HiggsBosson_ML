
# -*- coding: utf-8 -*-
import csv
import numpy as np
from tqdm import tqdm
from helpers import *

########## Functions to implement ########

def least_squares_GD(y, tx, initial_w, max_iters, gamma) :
    """Linear regression using gradient descent.

    Parameters
    ----------
    y : ndarray
        Labels.
    tx : ndarray
        Train features.
    initial_w : ndarray
        Initial weight vector.
    max_iters : int
         Number of steps to run.
    gamma :
        Step-size.

    Returns
    -------
    w :
        weight vector of the method
    loss :
        Corresponding loss value (cost function)

    """

    # Define parameters to store w and loss
    w = initial_w

    for n_iter in range(max_iters):
        # compute the gradient and the loss
        grad, e = gradient(y, tx, w)
        loss = mean_square_error(e)

        #Update the weight
        w = w - gamma * grad

    return w, loss

def least_squares_SGD(y, tx, initial_w, max_iters, gamma) :
    """Linear regression using stochastic gradient descent.

    Parameters
    ----------
    y : ndarray
        Labels.
    tx : ndarray
        Train features.
    initial_w : ndarray
        Initial weight vector.
    max_iters : int
         Number of steps to run.
    gamma :
        Step-size.

    Returns
    -------
    w :
        weight vector of the method
    loss :
        Corresponding loss value (cost function)

    """
    w = initial_w
    data_size = len(y)
    for n_iter in range(max_iters):
        # Using helper function form the lab
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):

            # compute a stochastic gradient
            grad, _ = gradient(y_batch, tx_batch, w)

            # update w through the stochastic gradient update
            w = w - gamma * grad

            # compute the loss
    loss = compute_loss(y, tx, w)

    return w, loss

def least_squares(y, tx):
    """SLeast squares regression using normal equations.

    Parameters
    ----------
    y : ndarray
        Labels.
    tx : ndarray
        Train features.

    Returns
    -------
    w :
        weight vector of the method
    loss :
        Corresponding loss value (cost function)

    """

    # Compute the weight vector
    w_sol = np.linalg.inv(tx.T.dot(tx)).dot(tx.T).dot(y)

    # Compute the loss
    loss = compute_loss(y, tx, w_sol)

    return w_sol, loss

def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations.

    Parameters
    ----------
    y : ndarray
        Labels.
    tx : ndarray
        Train features.
    lambda_ :
        Regularization parameter.

    Returns
    -------
    w :
        weight vector of the method
    loss :
        Corresponding loss value (cost function)

    """

    # Compute the bias terme
    a = tx.T.dot(tx) + 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    b = tx.T.dot(y)
    w_sol = np.linalg.inv(a).dot(b)

    # Can also use np.linalg.solve(a, b)
    # Check the most efficient
    loss = compute_loss(y, tx, w_sol)
    return w_sol, loss

def logistic_regression(y, tx, initial_w, max_iters= 10, gamma = 0.001):
    """Logistic regression using gradient descent.

    Parameters
    ----------
    y : ndarray
        Labels.
    tx : ndarray
        Train features.
    initial_w : ndarray
        Initial weight vector.
    max_iters : int
         Number of steps to run.
    gamma :
        Step-size.

    Returns
    -------
    w :
        weight vector of the method
    loss :
        Corresponding loss value (cost function)

    """

    # Replace -1 with 0 to perform logistic regression
    y_logist = np.copy(y)
    y_logist[np.where(y_logist == -1)] = 0

    w_sol = initial_w
    for iter in range(max_iters):
        # Compute gradient
        gradient = logistic_gradient(y_logist, tx, w_sol)
        # Compute the weight
        w_sol -= gamma*gradient

        #Compute the loss
    loss = logistic_loss(y_logist, tx, w_sol)

    return w_sol, loss

def reg_logistic_regression(y, tx, lambda_ , initial_w, max_iters = 10, gamma = 0.001):
    """Regularized logistic regression using gradient descent.

    Parameters
    ----------
    y : ndarray
        Labels.
    tx : ndarray
        Train features.
    lambda_ :
        Regularization parameter.
    initial_w : ndarray
        Initial weight vector.
    max_iters : int
         Number of steps to run.
    gamma :
        Step-size.

    Returns
    -------
    w :
        weight vector of the method
    loss :
        Corresponding loss value (cost function)

    """

    # Replace -1 with 0 to perform logistic regression
    y_logist = np.copy(y)
    y_logist[np.where(y_logist == -1)] = 0

    w_sol = initial_w
    for iter in range(max_iters):
        # Compute gradient
        gradient = logistic_gradient(y_logist, tx, w_sol) + 2 * lambda_ * w_sol
        # Compute the weight
        w_sol -= gamma*gradient

    loss = logistic_loss(y_logist, tx, w_sol) + lambda_ * np.squeeze(w_sol.T.dot(w_sol))

    return w_sol, loss











######### OTHER FUNCTIONS

def cross_validation(y, x, k_indices, k, lambda_, degree=1, func = "ridge",  **kwargs):
    """Perform first step of cross validation for the given function, default = ridge.
        This function does not loop across the k intervals or lambdas, it is a helper
        method for cross_validation_loop. It performs data splitting and applies the input
        regression to the data.

    Parameters
    ----------
    y : ndarray
        Labels.
    x : ndarray
        Train features..
    k_indices : ndarray
        cross-validation indices produced by applying build_k_indices function.
    k : int
        index of k_indices to use for current data split.
    lambda_ :
        Regularization parameter.
    degree : type
        polynomial degree of input data.
    func : string
        regression function to use - only use "ridge" or "reg_logist".
    **kwargs : dict
        args for 'func' parameter.

    Returns
    -------
    loss_tr: float
        training loss for current data split
    loss_te: float
        testing loss for current data split
    w: ndarray
        final feature vector
    """

    # K group in train, other in test
    te_ind = k_indices[k]

    #  k-1 split for training
    tr_ind = [i for i in range(k_indices.shape[0]) if i is not k]
    tr_ind = k_indices[tr_ind,:].reshape(-1)

    #Build the x and y train and test set
    y_tr = y[tr_ind]
    y_te = y[te_ind]

    x_tr = x[tr_ind, :]
    x_te = x[te_ind, :]

    # form data with polynomial degree, 1 oterwise
    tx_tr = build_polynomial(x_tr, degree)
    tx_te = build_polynomial(x_te, degree)

    # Apply function
    if func == "reg_logist" :
        w, loss_tr = reg_logistic_regression(y_tr, tx_tr, lambda_ , kwargs.get("initial_w"), kwargs.get("max_iters") , kwargs.get("gamma"))
        # calculate the loss for test data
        loss_te = logistic_loss(y_te, tx_te, w)
    elif func == "ridge" :
        w, loss_tr = ridge_regression(y_tr, tx_tr, lambda_)
        # calculate the loss for test data
        loss_te = compute_loss(y_te, tx_te, w)

    else: # in case of typo
        raise ValueError("Unknown function: " + func)

    return loss_tr, loss_te, w


def cross_validation_loop(y, x, k_fold, seed, lambdas, degree=1, func = 'ridge', **kwargs):
    """Performs full cross-validation by using the helper function 'cross_validation'. This function
        loops over all k data intervals and all input lambdas and averages out training and testing
        RMSE values. This function is only used for Ridge and Regularized Logistic Regression.

    Parameters
    ----------
    y : ndarray
        Labels.
    x : ndarray
        Train features..
    k_indices : ndarray
        cross-validation indices produced by applying build_k_indices function.
    k : int
        index of k_indices to use for current data split.
    lambdas: array
        Array of regularization parameters
    degree : type
        polynomial degree of input data.
    func : string
        regression function to use - only use "ridge" or "reg_logist".
    **kwargs : dict
        args for 'func' parameter.

    Returns
    -------
    rmse_tr: array
        Array of average training RMSE values for each lambda
    rmse_te: array
        Array of average testing RMSE values for each lambda
    """

    k_indices = build_k_indices(y, k_fold, seed)
    rmse_tr = []
    rmse_te = []

    for lambda_ in lambdas:
        rmse_tr_acc = 0 # accumulators
        rmse_te_acc = 0

        for k in range(k_fold):

            # get k-th RMSE values
            mse_tr_k, mse_te_k, _ = cross_validation(y, x, k_indices, k, lambda_, degree, func, **kwargs)
            rmse_tr_k = np.sqrt(2 * mse_tr_k) # calculate RMSE
            rmse_te_k = np.sqrt(2 * mse_te_k)

            # add to accumulators
            rmse_tr_acc += rmse_tr_k
            rmse_te_acc += rmse_te_k

        rmse_tr.append(rmse_tr_acc / k_fold) # append average rmse values
        rmse_te.append(rmse_te_acc / k_fold)

    return rmse_tr, rmse_te

def custom_cross_validation(y, x, k_fold, seed, regression, loss_func, *args):
    """Performs cross-validation on regression functions that do not have a lambda_ parameter.
        This function splits the data into k_fold parts and iterates across the parts, changing
        test and training data each time. At each iteration, it calculates the train and test RMSE
        values, then finally it averages them across iterations and returns these average values.

    Parameters
    ----------
    y : ndarray
        Labels.
    x : ndarray
        Train features.
    k_fold : type
        Description of parameter `k_fold`.
    seed : int
        Random Seed
    regression :
        Regression function to use.
    loss_func :
        Loss function to use.
    *args: array
        Array of args to feed to regression function
    Returns
    -------
    rmse_tr_avg: float
        average RMSE of training data
    rmse_te_avg: float
        average RMSE of test data

    """

    loss_tr_acc = 0
    loss_te_acc = 0
    for k in range(k_fold):
        y_tr, tx_tr, y_te, tx_te = split_data(y, x, k_fold, seed, k)

        w, loss_tr = regression(y_tr, tx_tr, *args)
        loss_te = loss_func(y_te, tx_te, w)

        loss_tr_acc += loss_tr
        loss_te_acc += loss_te

    loss_tr_avg = loss_tr_acc / k_fold
    loss_te_avg = loss_te_acc / k_fold

    rmse_tr_avg = np.sqrt(2*loss_tr_avg)
    rmse_te_avg = np.sqrt(2*loss_te_avg)

    return rmse_tr_avg, rmse_te_avg



def ridge_regression_final(degrees, y_tr, x_tr):
    lambdas = np.logspace(-5, -3, 20)
    deg = []
    lamb = []
    for degree in tqdm(degrees):
        rmse_tr, rmse_te = cross_validation_loop(y_tr, x_tr, 4, 1, lambdas, degree)
        lbd = lambdas[np.argmin(rmse_te)]
        deg.append(np.min(rmse_te))
        lamb.append(lbd)
    print("Best degree {0}, \n Best lambda for this degree : {1}".format(degrees[np.argmin(deg)], lamb[np.argmin(deg)]))
    return degrees[np.argmin(deg)], lamb[np.argmin(deg)]
    #return rmse_tr, rmse_te, lambdas
