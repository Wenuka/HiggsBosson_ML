import numpy as np
import csv
from implementations import *
from helpers import *


def main() :
    print ("Reading the training data..")
    DATA_TRAIN_PATH = '../data/train.csv'
    y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

    print ("Reading the testing data..")
    DATA_TEST_PATH = '../data/test.csv'
    _, tX_test, ids_test = load_csv_data(DATA_TEST_PATH) # TEST DATA HAS NO Y VALUES!!

    # Devide test and train according to jet number and clean
    print ("Start deviding the data w.r.t the jet number...")
    X, X_test, Y, ind, ind_test = devide_jet(tX, tX_test, y, ids, ids_test)
    # Make predictions for each group
    print ("Starting predictions...")
    pred={}
    for i in range(0,4):
        print("Jet number {0}".format(i))
        x = X[i]
        print ("  -Running Ridge Regression for 13 degrees to find best degree and lambda...")
        deg, lamb = ridge_regression_final(np.arange(1,14), Y[i], x)
        print ("  -Building the Polynomial...")
        x = build_polynomial(X[i], deg)
        print ("  -Finding the weights and the loss...")
        w, loss = ridge_regression(Y[i],x, lamb)
        print ("  -Predicting...")
        pred[i] = predict_labels(w, build_polynomial(X_test[i], deg), logistic = False)

    #Combine the predictions
    print ("Combining the results...")
    predic = combine(ind_test, pred)

    print ("Saving the results...")
    OUTPUT_PATH = '../data/best_predictions.csv'
    create_csv_submission(ids_test, predic, OUTPUT_PATH)
    print ("Done.")


if __name__ == '__main__':
    main()
