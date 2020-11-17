import numpy as np


def train_no_context(X, y, dH, lrate=1e-5, nepoch=200):
    ''' Trains a single-context CBOW extractor
    Args:
        X (numpy.ndarray): one-hot-encoding of data
        y (numpy.ndarray): expected labels
        dH (int): hidden layer dimension
        lrate (float): learning rate, defaults to 1e-5
        nepoch (int): number of training epochs, defaults to 200
    Returns:
        w1, w2: the weights of the neural network
    '''
    dIn, dH, dOut = X.shape[0], dH, X.shape[0]
    w1 = np.random.randn(dIn, dH)
    w2 = np.random.randn(dH, dOut)

    for epoch in range(nepoch):
        l1_out = X @ w1
        ypred = l1_out @ w2

        loss = np.square(y - ypred).sum()
        print(f"{epoch + 1} loss = {loss}")

        de_do = 2 * (y - ypred)
        grad_w1 = de_do @ l1_out
        w1 += lrate * grad_w1
    return w1, w2


def predict(X, w1, w2):
    l1_out = X @ w1
    ypred = l1_out @ w2
    return ypred
