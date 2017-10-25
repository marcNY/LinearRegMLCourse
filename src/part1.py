import numpy as np

from common import get_input_data, get_vectors


def compute_wrr(X, Y, lbda):
    n, d = X.shape
    I = np.identity(d)
    L = np.dot(lbda, I) + np.dot(X.transpose(), X)
    L_1 = np.linalg.inv(L)
    w_rr = L_1.dot(X.transpose()).dot(Y)
    return w_rr


def output_results(w_rr, lbda):
    output_file = 'wRR_%d.csv' % lbda
    np.savetxt(output_file, w_rr, fmt='%.10f', delimiter=",")
    return output_file


def run_part1(sysargs):
    # Get The input Arguments
    input_args = get_input_data(sysargs)
    print input_args
    X_train, Y_train, X_test = get_vectors(input_args)
    [n, d] = X_train.shape
    print("X_train shape:%s" % str(X_train.shape))
    print("Y_train shape:%s" % str(Y_train.shape))
    print("X_test shape:%s" % str(X_test.shape))

    w_rr = compute_wrr(X_train, Y_train, input_args['lambda'])
    print "w_rr computed%s" % str(w_rr)
    print "w_rr shape is %s" % str(w_rr.reshape(1, d).shape)
    print "w_rr computed%s" % str(w_rr)

    output_results(w_rr, input_args['lambda'])
    pass
