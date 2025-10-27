import numpy as np

def perceptron(X, weights, bias):
    linear_output = np.dot(X, weights) + bias
    activated_output = np.where(linear_output >= 0, 1, 0)
    return activated_output


