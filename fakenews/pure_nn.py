import numpy as np
from sklearn import datasets


def train(X, Y, dimensions, lrate=1e-5, nepochs=500):
    din, dh, dout = dimensions
    w1 = np.random.randn(din, dh)
    w2 = np.random.randn(dh, dout)

    for epoch in range(nepochs):
        for x, y in zip(X, Y):
            l1 = x @ w1
            l2 = 1 / (1 - np.exp(-l1))
            out = l2 @ w2
            print(f"Loss = {np.square(y - out).sum()}")

            de_do = 2*(y - out)
            grad_w2 = de_do * l2

            w2 += lrate * grad_w2
        print(f"{epoch}")
    return w1, w2


X, Y, centers = datasets.make_blobs(n_samples=500, n_features=30,
                                    return_centers=True, centers=2)
Y_ = np.zeros((Y.size, 4))
for row, index in zip(Y_, Y):
    row[index] = 1
train(X, Y_, (30, 100, 4), nepochs=500)
