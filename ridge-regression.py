import numpy as np


class Regression():

    def __init__(self):
        pass

    def rmse(self, pred, label):
        return np.sqrt(np.mean(np.square(pred - label)))

    def construct_polynomial_feats(self, x, degree):
        return np.power(x.reshape(len(x), 1), np.arange(degree + 1))

    def predict(self, xtest, weight):
        return np.matmul(xtest, weight)

    # closed form
    def ridge_fit_closed(self, xtrain, ytrain, c_lambda):  # [5pts]
        n, d = xtrain.shape
        idy = np.eye(d)
        idy[0, 0] = 0
        pseudo_inv = np.linalg.pinv((xtrain.T @ xtrain) + (c_lambda * idy))
        return pseudo_inv @ xtrain.T @ ytrain

    #gradient descent
    def ridge_fit_GD(self, xtrain, ytrain, c_lambda, epochs=500, learning_rate=1e-7):
        weight = np.zeros((xtrain.shape[1], 1))
        for i in range(epochs):
            n = learning_rate / xtrain.shape[0]
            xweight = xtrain @ weight - ytrain
            m = xtrain.T @ xweight + c_lambda * weight
            d = n * m
            weight -= d
        return weight

    #stochastic gradient descent
    def ridge_fit_SGD(self, xtrain, ytrain, c_lambda, epochs=100, learning_rate=0.001):
        weight = np.zeros((xtrain.shape[1], 1))
        for i in range(epochs):
            for j in range(0, xtrain.shape[0]):
                p = xtrain[j]
                point = p.reshape(1, xtrain.shape[1])
                pweight = point @ weight
                y = pweight - ytrain[j]
                m = point.T @ y + c_lambda * weight
                d = learning_rate * m
                weight -= d
        return weight
