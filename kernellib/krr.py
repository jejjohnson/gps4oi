import numpy as np
from kernellib.utils import cholesky_solve
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.linalg import solve_triangular


class KRR(RegressorMixin, BaseEstimator):
    def __init__(self, alpha=0.1, kernel=None):
        self.alpha = alpha
        self.kernel = kernel

    def fit(self, X, y):

        # create kernel matrix (data)
        K = self.kernel(X)

        # add the noise to diagonal elements
        K[np.diag_indices_from(K)] += self.alpha
        # np.fill_diagonal(K, K.diagonal() + self.alpha)

        # find the weights, alpha = (K + noise)^-1 y
        # L = cholesky(K + sigma^2 I
        # alpha = L^T \ (L \ y)
        weights, L = cholesky_solve(K, y)

        self.weights_ = weights
        self.L_ = L
        self.X_fit_ = X

        return self

    def predict(self, X, return_var: bool = False, return_cov: bool = False):
        # TODO include noise

        # make predictions
        K_trans = self.kernel(X, self.X_fit_)

        # predictive mean
        y_pred = np.dot(K_trans, self.weights_)

        if return_var:
            # TODO Fix this to the kernel

            # diagonal elements
            y_var = self.kernel.diag(X)

            # solve
            v = solve_triangular(self.L_[0], K_trans.T, lower=True)
            y_var = y_var - np.einsum("ij,ji->i", v.T, v)

            return y_pred, y_var

        elif return_cov:

            K_star = self.kernel(X)
            v = solve_triangular(self.L_[0], K_trans.T, lower=True)
            y_cov = K_star - v.T @ v

            return y_pred, y_cov
        else:

            return y_pred

    def sample(self, X: np.ndarray = None, n_samples: int = 1, seed: int = 123):

        rng = np.random.RandomState(seed)

        # predictions
        if X is None:
            X = self.X_fit_

        # make predictions (mean, cov)
        y_mu, y_cov = self.predict(X=X, return_cov=True, return_var=False)

        # sample from a Gaussian distribution
        y_samples = rng.multivariate_normal(y_mu.squeeze(), y_cov, n_samples).T

        return y_samples
