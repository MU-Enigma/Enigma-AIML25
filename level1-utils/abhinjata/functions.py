import numpy as np
import pandas as pd
import math
from queue import PriorityQueue
from typing import Callable

# I. Activation Functions 

def relu(x): 
    return np.maximum(0, x)

def relu_derivative(x):
    #need this for grad desc
    return np.where(x > 0, 1, 0)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def softmax(x: list[float]) -> list[float]:

    # exponentiate all elements
    exps = [math.exp(i) for i in x]
    
    # get the sum of all exponentiated elements
    sum_of_exps = sum(exps)
    
    # normalize 
    probabilities = [j / sum_of_exps for j in exps]
    
    return probabilities

def tanh(x):

    e_pos = math.exp(x) #e^x
    e_neg = math.exp(-x) #e^-x
    
    return (e_pos - e_neg) / (e_pos + e_neg)


# II. Crucial Dimensional Operations 

def dotProd(v, u):

    if len(v) != len(u):
        raise ValueError("Vectors must be the same length to compute the dot product.")

    # Use zip to pair corresponding elements (v1, u1), (v2, u2) -> list of tuples
    return sum(v_i * u_i for v_i, u_i in zip(v, u))


def cosineSim(v, u):

    """
    Similarity = (v · w) / (||v|| * ||w||)
    
    Inputs: v (List[float]): The first vector, w (List[float]): The second vector
    Output: float -> The cosine similarity, a value between -1 and 1

    """
    #numerator
    numerator = dotProd(v, u)
    
    #denominator
    mag_v = math.sqrt(dotProd(v,v))
    mag_u = math.sqrt(dotProd(u,u))
    
    #edge case
    if mag_v == 0 or mag_u == 0:
        return 0.0
    
    return numerator / (mag_v * mag_u)

# III. Normalization Functions 

def manhattanNorm(v: list[float]) -> list[float]:

    #scales vector so the sum of items is 1 

    l1Norm = sum(abs(x) for x in v)

    #edge case
    if l1Norm == 0:
        return v

    return[ x / l1Norm for x in v]

def euclidNorm(v : list[float]) -> list[float]:

    #uses euclian length to scale the length to 1

    #magnitude of eucliena length
    l2_norm = math.sqrt(dotProd(v, v))
    
    #edge case
    if l2_norm == 0:
        return v
        
    return [x / l2_norm for x in v]

def minmaxScale(v: list[float]) -> list[float]:
    
    # Applies Min-Max scaling to a list of feature values (a column)
    # Scales the values to be in the range [0, 1].

    minVal = min(v)
    maxVal = max(v)
    
    #Range
    dataRange = maxVal - minVal
    
    #Edge case
    if dataRange == 0:
        return [0.0] * len(v)
        
    #(x - min) / (max - min)
    return [(x - minVal) / dataRange for x in v]

def aStarSearch(graph: dict[str, list[tuple[str, float]]], start: str, goal: str, heuristic: Callable[[str, str], float]) -> tuple[list[str], float]:
    
    """
    Finds the shortest path from start to goal in a graph using A* search.
    This is absolutely how you go off on a massive tangent before you DAA minor 2 and learn Djikstra indirectly while contributing to hacktober
    """
    
    # 1. Initialization
    visitedArray = PriorityQueue()
    visitedArray.put((0, start))
    
    backtrackItem: dict[str, str] = {}
    
    gScore: dict[str, float] = {node: float('inf') for node in graph}
    gScore[start] = 0
    
    fScore: dict[str, float] = {node: float('inf') for node in graph}

    # Here is where the heuristic is first used its all djikstra init so far
    fScore[start] = heuristic(start, goal)
    
    visitedArray_hash = {start}

    # 2. Main Search Loop
    while not visitedArray.empty():
        currentF, current = visitedArray.get()
        visitedArray_hash.remove(current)

        if current == goal:
            path = reconstruct_path(backtrackItem, current)
            return path, gScore[current]

        # Relaxation Logic
        for neighbor, cost in graph[current]:
            tentative_gScore = gScore[current] + cost
            
            if tentative_gScore < gScore[neighbor]:
                backtrackItem[neighbor] = current
                gScore[neighbor] = tentative_gScore
                
                # Here is where the heuristic is used every step
                fScore[neighbor] = tentative_gScore + heuristic(neighbor, goal)
                
                if neighbor not in visitedArray_hash:
                    visitedArray.put((fScore[neighbor], neighbor))
                    visitedArray_hash.add(neighbor)
    
    # 3. No Path Found
    return [], 0.0

def reconstruct_path(backtrackItem: dict[str, str], current: str) -> list[str]:

    #Helper function to rebuild the path from the backtrackItem map 

    total_path = [current]

    while current in backtrackItem:

        current = backtrackItem[current]
        total_path.append(current)

    return total_path[::-1]


#Courtesy: Gemini 2.5 Pro -> for general learning and major control logic of A Star Search
