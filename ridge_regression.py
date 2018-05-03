import numpy as np
from scipy.sparse import csr_matrix

class Ridge():
    def __init__(self, max_iters=2000, alpha=0.1, lmd=0.1):
        self.max_iters = max_iters
        self.alpha = alpha
        self.lmd = lmd

    # def compute_cost(self, X, y, w, lmd):
    #     return (1. / (2. * X.shape[0])) * (np.sum((np.dot(X, w) - y) ** 2.) + lmd * np.dot(w.T, w))

    def gradient_descent(self, X, y, w, max_iters, alpha, lmd):
        # cost = np.zeros((max_iters, 1))
        for i in range(max_iters):
            # cost[i] = self.compute_cost(X, y, w, lmd)
            w = w - (alpha / X.shape[0]) * (X.T.dot((X.dot(w) - y)) + lmd * w)
        return w

    def regression(self, X, y):
        Xn = X
        yn = np.ndarray.copy(y)
        w = np.zeros((Xn.shape[1] + 1, 1))

        self.X_mean = np.mean(Xn, axis=0)
        self.X_std = np.std(Xn.todense(), axis=0)
        Xn -= self.X_mean
        self.X_std[self.X_std == 0] = 1
        Xn /= self.X_std
        Xn = np.hstack((np.ones(Xn.shape[0])[np.newaxis].T, Xn))
        Xn = csr_matrix(Xn)
        self.y_mean = yn.mean(axis=0)
        yn -= self.y_mean

        self.w= self.gradient_descent(Xn, yn, w, self.max_iters, self.alpha, self.lmd)

    def predict(self, X):
        Xn = X

        Xn -= self.X_mean
        Xn /= self.X_std
        Xn = np.hstack((np.ones(Xn.shape[0])[np.newaxis].T, Xn))

        return Xn.dot(self.w) + self.y_mean
