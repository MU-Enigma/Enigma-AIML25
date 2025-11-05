import math
import numpy as np

# Sigmoid Function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# ReLU function 
def relu(x):
    return max(0, x)

# tanh function 
def tanh(x):
    return math.tanh(x)

#vector dot product
def vector_dot_product(x,y):
    return x@y

#cosine similarity
def cosine_similarity(vec1, vec2):

    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    similarity = dot_product / (norm_vec1 * norm_vec2)
    return similarity

#l1 normalization
def l1_normalize(x):
    return x / np.sum(np.abs(x))

#l2 normalization
def l2_normalize(x):
    return x / np.sqrt(np.sum(x**2))

#min-max normalization
def min_max_normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

#Heuristic function
''' This heuristic measures how far each tile is from its goal position (up/down/left/right steps).
    The smaller the total distance → the closer the grid is to the goal.
    Still very easy to understand, just one level deeper than counting mismatches.'''
def heuristic(current, goal):
    distance = 0
    for i in range(3):
        for j in range(3):
            for x in range(3):
                for y in range(3):
                    if current[i][j] == goal[x][y]:
                        distance += abs(i - x) + abs(j - y)
    return distance
