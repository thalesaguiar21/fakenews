''' This module contains several artificiall intelligence models
implemented with numpy or pandas only
'''
import numpy as np
from sklearn import datasets


def train(X, y, dimensions, lrate=1e-5, nepochs=100):
    din, dh, dout = dimensions
    w1 = np.random.randn(din, dh)
    w2 = np.random.randn(dh, dout)

    for epoch in range(nepochs):
        # forward pass
        hin = X @ w1
        hout = 1 / (1+np.exp(-hin))
        ypred = hout @ w2

        loss = np.square(y - ypred).sum()
        print(f"{epoch + 1} loss = {loss}")

        # backward pass
        de_do = 2 * (y - ypred)
        grad_w2 = hout.T @ de_do

        de_dh = de_do @ w2.T
        de_dsig = de_dh * hout*(1 - hout)
        grad_w1 = X.T @ de_dsig

        w2 += lrate * grad_w2
        w1 += lrate * grad_w1
    return w1, w2


def pred(X, *weights):
    w1, w2 = weights
    # forward pass
    hin = X @ w1
    hout = 1 / (1+np.exp(-hin))
    return hout @ w2


X, y, centers = datasets.make_blobs(n_samples=500, n_features=30,
                                    return_centers=True, centers=2)
y_ = np.zeros((y.size, 4))
for row, index in zip(y_, y):
    row[index] = 1
wa, wb = train(X, y_, (30, 100, 4), nepochs=500)
ypred = pred(X, wa, wb)
np.set_printoptions(precision=3)
print(ypred[:3])
print(y_[:3])
