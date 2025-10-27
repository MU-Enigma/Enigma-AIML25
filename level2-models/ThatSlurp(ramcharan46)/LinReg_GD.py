import numpy as np

def predict(X, m, c):
    return m * X + c

def mean_squared_error(Y_true, Y_pred):
    N = len(Y_true)
    loss = (1 / (2 * N)) * np.sum((Y_true - Y_pred) ** 2)
    return loss

def gradient_descent(X, Y, m, c, learning_rate):
    N = len(X)
    Y_pred = predict(X, m, c)
    error = Y_pred - Y
    dm = (1 / N) * np.sum(error * X)
    dc = (1 / N) * np.sum(error)
    m_new = m - learning_rate * dm
    c_new = c - learning_rate * dc
    return m_new, c_new, dm, dc


