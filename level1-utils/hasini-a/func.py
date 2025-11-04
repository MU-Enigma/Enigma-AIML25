import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def relu(x):
    return max(0, x)

def tanh(x):
    return math.tanh(x)

def dot_product(v1, v2):
    if len(v1) != len(v2):
        raise ValueError("Vectors must be of the same length")
    return sum(a * b for a, b in zip(v1, v2))

def cosine_similarity(v1, v2):
    dot = dot_product(v1, v2)
    magnitude_v1 = math.sqrt(dot_product(v1, v1))
    magnitude_v2 = math.sqrt(dot_product(v2, v2))
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0
    return dot / (magnitude_v1 * magnitude_v2)

def l1_normalize(v):
    norm = sum(abs(x) for x in v)
    return [x / norm for x in v] if norm != 0 else v

def l2_normalize(v):
    norm = math.sqrt(sum(x ** 2 for x in v))
    return [x / norm for x in v] if norm != 0 else v

def min_max_normalize(v):
    min_val = min(v)
    max_val = max(v)
    return [(x - min_val) / (max_val - min_val) if max_val != min_val else 0 for x in v]

def heuristic_function(v, target):
    return sum((x - target) ** 2 for x in v)

v1 = [1, 2, 3]
v2 = [4, 5, 6]

print(sigmoid(0))
print(relu(-3))
print(tanh(1))

print(dot_product(v1, v2))
print(cosine_similarity(v1, v2))

print(l1_normalize(v1))
print(l2_normalize(v1))
print(min_max_normalize(v1))

target_value = 5
print(heuristic_function(v1, target_value))
