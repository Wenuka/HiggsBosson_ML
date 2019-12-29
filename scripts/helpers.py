# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np
#from implementations import *


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids

def predict_label_multiset(weights, data,meanArry,stdArry):
    """Generates class predictions given weights, and a test data matrix"""
#     limit = 0
#     lower = -1
#     upper = 1
    y_pred = np.zeros(np.shape(data)[0])
    for index_, dd in enumerate(data):
        print ("Analysing...",index_)
        if dd[0] == -999:
            newVal = (np.delete(dd,[0,4,5,6,12,23,24,25,26,27,28]) - meanArry[1])/ stdArry[1]
            y_pred[index_] = (np.dot( weights[1] ,np.append([1],newVal )) )
        elif dd[23] == -999:
            newVal = (np.delete(dd,[4,5,6,12,22,23,24,25,26,27,28,29]) - meanArry[2])/ stdArry[2]
            y_pred[index_] = (np.dot( weights[2] ,np.append([2],newVal )) )
        elif dd[4] == -999:
            newVal = (np.delete(dd,[4,5,6,12,22,26,27,28]) - meanArry[3])/ stdArry[3]
            y_pred[index_] = (np.dot( weights[3] ,np.append([3],newVal )) )
        else:
            newVal = (dd - meanArry[0])/ stdArry[0]
            y_pred[index_] = (np.dot( weights[0] ,np.append([0],newVal )) )




#     y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1

    return y_pred

def predict_labels(weights, data, logistic = False):
    """Generates class predictions given weights, and a test data matrix"""
    limit = 0.5 if logistic else 0
    lower = 0 if logistic else -1
    upper = 1

    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= limit)] = lower
    y_pred[np.where(y_pred > limit)] = upper

    return y_pred

def compare_predictions(y_true, y_pred):
    """
    Compares y_true to y_pred and checks how many elements are equal.
    """
    n_equal = np.sum(y_true == y_pred)
    perc = n_equal/len(y_true) * 100
    print("# equal elements: {0} i.e. {1} %".format(n_equal, perc))

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})
    csvfile.close()



def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def cross_validation_visualization(lambds, mse_tr, mse_te):
    """visualization the curves of mse_tr and mse_te."""
    plt.semilogx(lambds, mse_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambds, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("rmse")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")


######### Helpers for cleaning and model selection #############


def remove_cols(data):
    """Clean input data by removing columns that contain 100% of -999 entries.

    Parameters
    ----------
    data : ndarray
        A numpy 2D array containing the data we want to remove the -999 entries.

    Returns
    -------
    new_data : ndarray
        A numpy 2D array with the columns wit 100%  of -99 removed.

    """
    # Find the -999 entries
    cols = np.where(data == -999)[1]
    unique, count = np.unique(cols, return_counts = True)

    N = data.shape[0]
    d = dict(zip(unique, count))
    # Printing to know the %age of 100% -999

    print("column\t\t-999 count\t% total")
    for col in d:
        print("{0}\t\t{1}\t\t{2}".format(col, d[col], d[col] / N * 100))
        if (d[col] / N) < 1 :
            # Remove columns with 100% of -999
            unique = np.delete(unique, col)

    return np.delete(data, unique, axis=1)

def clean(tX):
    """Clean the data by :
            - Replacing -999 values with median of the vector
            - Removing outliers .

    Parameters
    ----------
    tX : ndarray
        Feature array .

    Returns
    -------
    tX : ndarray
        Cleaned feature array.

    """

    # Replace -999 with median
    tX_rm = tX.copy()

    #Nan values to replace
    inds = np.where(tX_rm == -999.)
    #Replace -999 with nan
    tX_rm[inds] = np.nan

    # Compute median
    med = np.nanmedian(tX_rm, axis=0)
    #Replaces nan with median
    tX_rm[inds] = np.take(med, inds[1])

    # Remove outliers

    # Compute the quantiles
    Q1 = np.quantile(tX_rm, 0.25, axis=0)
    Q3 = np.quantile(tX_rm, 0.75, axis=0)

    # Define the min and max values for outliers
    out_max = 2.5*Q3-1.5*Q1
    out_min = 2.5*Q1-1.5*Q3

    #Remove high values
    inds = np.where(tX_rm > out_max)
    tX_rm[inds] = np.take(out_max, inds[1])

    #Remove low values
    inds = np.where(tX_rm < out_min)
    tX_rm[inds] = np.take(out_min, inds[1])
    return tX_rm

def preprocess(X, poly = False):
    """Preprocess the data by :
            - Removing columns with 100% of -999
            - Cleaning the data (please reffer to the clean method)
            - Remove columns with zero mean (in practice, columns with only zeros).
            - Build a polynomial if needed

    Parameters
    ----------
    X : ndarray
        The features to preprocess
    poly : bolean
        If a 1 column is added at he beggining of the array

    Returns
    -------
    X : ndarray
        The pre-processed feature array.

    """

    X_rm = remove_cols(X)
    X_rm = clean(X_rm)

    mean = np.mean(X_rm, axis= 0)
    std = np.std(X_rm, axis=0)
    # Remove zero mean
    cols = np.where(mean == 0)
    dataX_rm = np.delete(X_rm, cols, axis=1)
    dataX_rm, _, __ = standardize(dataX_rm)
    return build_polynomial(dataX_rm, 1) if poly else dataX_rm

def devide_jet(tX, tX_test, y, ids, ids_test):
    """Split the data in 4 groups according to the jet number and preprocess de data (Standardization, cleaning, removing outliers)

    Parameters
    ----------
    tX : ndarray
        Train features to devide.
    tX_test : ndarray
        Test feature vector to devide.
    y : ndarray
        Train label vector to devide.
    ids : ndarray
        Indices of the train data .
    ids_test : ndarray
        Indices fo the test data.

    Returns
    -------
    X : dict
        Dictionary containing 4 ndarrays coresponding to devides train features
    X_test : dict
        Dictionary containing 4 ndarrays coresponding to devides test features
    Y : dict
        Dictionary containing 4 ndarrays coresponding to devides train labels
    ind : dict
        Dictionary containing 4 ndarrays coresponding to devides train indices
    ind_test : dict
        Dictionary containing 4 ndarrays coresponding to devides test indices

    """
    # Devide values for diferent jets
    X ={}
    X_test = {}
    Y = {}
    ind = {}
    ind_test = {}
    #For each jet numbers
    for i in range(0,4):
        # Jet index
        j = tX[:, 22] == i
        jet =  tX[j]
        jet = np.delete(jet, 22, axis = 1)

        #For test values
        j_test = tX_test[:, 22] == i
        jet_test = tX_test[j_test]
        jet_test = np.delete(jet_test, 22, axis = 1)

        #For y
        y_ = y[j]

        #For indices
        ids_ = ids[j]

        #For test indices
        ids_test_ = ids_test[j_test]

        #Preproces the train and test data
        print ("For the PRI_Jet_num_%i: \npreprocessing the training data..."%i)
        t_X = preprocess(jet, False)
        print ("preprocessing the testing data...")
        t_X_test = preprocess(jet_test, False)

        X[i] = t_X
        X_test[i] = t_X_test
        Y[i] = y_
        ind[i] = ids_
        ind_test[i] = ids_test_

    return X, X_test, Y, ind, ind_test

def combine(ids , y) :
    """Combine the 4 jet groups labels, according to their indices.

    Parameters
    ----------
    ids : dict
        Indices from the 4 jet groups
    y : dict
        Predicted labels.

    Returns
    -------
    pred :
        The combined prediction

    """
    ids_ = np.concatenate((ids[0], ids[1],ids[2],ids[3]))
    y_ = np.concatenate((y[0], y[1], y[2], y[3]))
    #stack = np.stack((ids_, y_), axis=1)
    return y_[np.argsort(ids_)]

#standardise data through each column
def standardize(x):
    """Perform data standardization .

    Parameters
    ----------
    x : ndarray
        Train features.

    Returns
    -------
    x : ndarray
        Standardized train features.
    mean :
        The mean  of the features
    std :
        The standard deviation of the features

    """

    mean = np.mean(x, axis=0)
    x = x - mean
    std = np.std(x, axis=0)
    x = x / std
    return x, mean, std

# Build polynomial from given code lab 4
def build_polynomial(x, degree):
    """Polynomial basis functions for input data x, for j=0 up to j=degree.

    Parameters
    ----------
    x : ndarray
        Train features.
    degree : int
        The degree for polynomial basis function.

    Returns
    -------
    poly :
        The polynomial basis function.

    """
    """"""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly

# Build k indices from given code Lab4
def build_k_indices(y, k_fold, seed):
    """Builds indices for k-fold cross-validation

    Parameters
    ----------
    y : ndarray
        Labels.
    k_fold : type
        Number of data intervals to use in cross-validation.
    seed : int
        Random Seed

    Returns
    -------
    ndarray
        Array of random equal-sized index intervals (2D array) to use for cross-validation.

    """
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def split_data(y, x, k_fold, seed, k):
    """Split the data randomly according to k_fold (e.g. k_fold = 4 will give a 75%-25% split)

    Parameters
    ----------
    y : ndarray
        Labels.
    x : ndarray
        Train features.
    k_fold : int
        split ratio indicator: the data is split into k_fold parts, of which k_fold-1 are used for training and 1 for testing.
    seed : int
        Random Seed
    k : int
        index of k_indices to use for training data.

    Returns
    -------
    y_tr: ndarray
        training data y vector
    x_tr: ndarray
        training data x matrix
    y_te: ndarray
        testing data y vector
    x_te: ndarray
        testing data x matrix

    """
    # K group in train, other in test
    k_indices = build_k_indices(y, k_fold, seed)
    te_ind = k_indices[k]

    #  k-1 split for training
    tr_ind = [i for i in range(k_indices.shape[0]) if i is not k]
    tr_ind = k_indices[tr_ind,:].reshape(-1)

    #Build the x and y train and test set
    y_tr = y[tr_ind]
    y_te = y[te_ind]

    x_tr = x[tr_ind, :]
    x_te = x[te_ind, :]

    return y_tr, x_tr, y_te, x_te


########### General Helpers #############

def mean_square_error(e):
    """calculates MSE for a given error vector

    Parameters
    ----------
    e : ndarray
        error vector.

    Returns
    -------
    float
        MSE for given e.

    """
    """Mean square error for vector e"""
    return 1/2 * np.mean(e**2)


def mean_absolute_error(e):
    """Short summary.

    Parameters
    ----------
    e : type
        Description of parameter `e`.

    Returns
    -------
    type
        Description of returned object.

    """
    """Mean absolute error for vector e."""
    return np.mean(np.abs(e))


def compute_loss(y, tx, w, mae = False):
    """Computes loss for given function

    Parameters
    ----------
    y : ndarray
        Labels.
    tx : ndarray
        Train features.
    w : ndarray
        Weight vector.
    mae : bolean
        Indicate if we want to compute the mean absolute error, defaul mean square.

    Returns
    -------
    float
        MSE or MAE loss

    """
    e = y - tx.dot(w)
    return mean_absolute_error(e) if mae else mean_square_error(e)

def logistic_loss(y, tx, w):
    """Compute loss (log likelihood with sigmoid)

    Parameters
    ----------
    y : ndarray
        Labels.
    tx : ndarray
        Train features.
    w : ndarray
        Weight vector.

    Returns
    -------
    float
        Loss using sigmoid log likelihood

    """
    txw = tx.dot(w)
    a = np.log(1+np.exp(txw))
    b = y*(txw)
    return np.sum(a-b)


def sigmoid(t):
    """Sigmoid function.

    Parameters
    ----------
    t : float
        exponent


    Returns
    -------
    s : float
        sigmoid function result
    """
    return 1.0 / (1 + np.exp(-t))

def gradient(y, tx, w):
    """Compute a gradient.

    Parameters
    ----------
    y : ndarray
        Labels.
    tx : ndarray
        Features.
    w : ndarray
        Weight vector.

    Returns
    -------
    grad :
        The computed gradient
    e :
        The error

    """
    #Conpute the error
    e = y - tx.dot(w)
    grad = -tx.T.dot(e) / len(e) # Compute the gradient

    return grad, e

def logistic_gradient(y, tx, w):
    """Compute the gradient for logistic regression .

    Parameters
    ----------
    y : ndarray
        Labels.
    tx : ndarray
        Features.
    w : ndarray
        Weight vector.

    Returns
    -------
    grad :
        The computed gradient

    """
    grad = tx.T.dot(sigmoid(tx.dot(w))-y)
    return grad
