import numpy as np


def get_input_data(datalist):
    output_dict = {}
    output_dict['lambda'] = float(datalist[1])
    output_dict['sigma'] = float(datalist[2])
    output_dict['x_train'] = (datalist[3])
    output_dict['y_train'] = (datalist[4])
    output_dict['x_test'] = (datalist[5])
    return output_dict


def get_vectors(inputargs):
    X_train = np.genfromtxt(inputargs['x_train'], delimiter=",")
    Y_train = np.genfromtxt(inputargs['y_train'], delimiter=",")
    X_test = np.genfromtxt(inputargs['x_test'], delimiter=",")
    return X_train, Y_train, X_test


def preprocess(X_train, Y_train, X_test):
    Y_train = Y_train - Y_train.mean()
    X_test = (X_test - X_test.mean()) / X_test.var()
    X_train = (X_train - X_train.mean()) / X_train.var()
    return X_train, Y_train, X_test
