import numpy as np

from common import get_input_data, get_vectors


def compute_mu(lbda, sigma, X, y):
    [n, d] = X.shape
    I = np.identity(d)
    den = lbda * sigma ^ 2 * I + np.dot(X.transpose(), X)
    num = np.dot(X.transpose(), y)
    mu = np.dot(np.linalg.inv(den), num)
    return mu, den, num


def compute_covar(lbda, sigma, X, y):
    [n, d] = X.shape
    I = np.identity(d)
    covarinv = lbda * I
    covarinv += (sigma ** -1) * np.dot(X.transpose(), X)
    covar = np.linalg.inv(covarinv)
    return covar, covarinv


def update_covar(covarinv, sigma, X0):
    covarinv = covarinv + sigma ** -1 * np.dot(X0, X0.transpose())
    covar = np.linalg.inv(covarinv)
    return covar, covarinv


def compute_vary(sigma, covar, x0):
    return sigma + x0.transpose().dot(covar).dot(x0)


def analyseorder(sigma, lbda, X, Y, X0M):
    n, d = X0M.shape
    covar, covarinv = compute_covar(lbda, sigma, X, Y)
    updatevary = lambda x: compute_vary(sigma, covar, x)
    sigy0 = np.apply_along_axis(updatevary, 1, X0M)
    output = sigy0.argsort()[::-1] + 1
    return output


def print_results(output, lbda, sigma):
    filename = 'active_%s_%s.csv' % (int(lbda), int(sigma))
    output = output.reshape(1, len(output))
    np.savetxt(filename, output, fmt='%d', delimiter=",")
    return filename


def run_part2(args):
    # Get The input Arguments
    input_args = get_input_data(args)
    print input_args
    X_train, Y_train, X_test = get_vectors(input_args)
    output = analyseorder(input_args['sigma'], input_args['lambda'], X_train, Y_train, X_test)
    print_results(output, input_args['lambda'], input_args['sigma'])
    pass
