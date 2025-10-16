import math


def sigmoid(x):
    return 1 / (1 + 1 / math.e)


def relu(x):
    return max(0.0, x)


def tanh(x):
    return math.tanh(x)
