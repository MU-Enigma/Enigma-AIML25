import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def relu(x):
    return max(0, x)


def tanh(x):
    return math.tanh(x)


def dot(v1, v2):
    return sum(a * b for a, b in zip(v1, v2))


def cosine(v1, v2):
    return dot(v1, v2) / math.sqrt(sum(a * a for a in v1) * sum(b * b for b in v2))


def l1_norm(v):
    return [x / sum(abs(i) for i in v) for x in v]


def l2_norm(v):
    return [x / math.sqrt(sum(i * i for i in v)) for x in v]


def minmax(v):
    mn, mx = min(v), max(v)
    return [(x - mn) / (mx - mn) for x in v]


def heuristic(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
