import math
import numpy as np
import functions as utils

if __name__ == "__main__":
    print("--- Starting Function Tests ---")

    # --- 1. Activation Functions ---
    print("\n--- I. Activation Functions ---")
    
    # Test relu & relu_derivative (using numpy as required)
    relu_input = np.array([-5.0, 0.0, 3.0, 10.0])
    print(f"relu({relu_input}) = {utils.relu(relu_input)}")
    print(f"relu_derivative({relu_input}) = {utils.relu_derivative(relu_input)}")
    
    # Test sigmoid (using numpy as required)
    sigmoid_input = np.array([-1.0, 0.0, 1.0])
    print(f"sigmoid({sigmoid_input}) = {utils.sigmoid(sigmoid_input)}")
    
    # Test softmax (using math/lists)
    softmax_input = [2.0, 1.0, 0.1]
    softmax_output = utils.softmax(softmax_input)
    print(f"softmax({softmax_input}) = {softmax_output}")
    print(f"Sum of softmax output: {sum(softmax_output)}") # Should be 1.0
    
    # Test tanh (using math)
    print(f"tanh(0) = {utils.tanh(0)}")
    print(f"tanh(1.5) = {utils.tanh(1.5)}")

    # --- 2. Crucial Dimensional Operations ---
    print("\n--- II. Crucial Dimensional Operations ---")
    
    v1 = [1, 2, 3]
    v2 = [4, 5, 6]
    v_origin = [0, 0]
    v_345 = [3, 4]
    v_ortho = [1, 0]
    v_ortho2 = [0, 1]
    v_opposite = [-1, -2, -3]
    
    print(f"dotProd({v1}, {v2}) = {utils.dotProd(v1, v2)}") # 1*4 + 2*5 + 3*6 = 32
    
    # Test cosineSim
    print(f"cosineSim({v1}, {v1}) = {utils.cosineSim(v1, v1)}") # 1.0
    print(f"cosineSim({v1}, {v_opposite}) = {utils.cosineSim(v1, v_opposite)}") # -1.0
    print(f"cosineSim({v_ortho}, {v_ortho2}) = {utils.cosineSim(v_ortho, v_ortho2)}") # 0.0
    
    # Test euclidean_distance
    print(f"euclidean_distance({v_origin}, {v_345}) = {utils.euclidean_distance(v_origin, v_345)}") # 5.0

    # --- 3. Normalization Functions ---
    print("\n--- III. Normalization Functions ---")
    
    norm_vec = [1, -2, 3] # L1 norm = 1+2+3 = 6
    print(f"manhattanNorm({norm_vec}) = {utils.manhattanNorm(norm_vec)}")
    
    norm_vec_2 = [3.0, 4.0] # L2 norm = 5.0
    print(f"euclidNorm({norm_vec_2}) = {utils.euclidNorm(norm_vec_2)}") # [0.6, 0.8]
    
    scale_vec = [10, 20, 30, 40, 50]
    print(f"minmaxScale({scale_vec}) = {utils.minmaxScale(scale_vec)}") # [0.0, 0.25, 0.5, 0.75, 1.0]

    # --- 4. A* Search Algorithm ---
    print("\n--- IV. A* Search Algorithm ---")
    
    # 1. Define a graph
    # (A) --1-- (B)
    #  | \       |
    #  5  1----- (C)
    #  | /       |
    # (D) --1-- (GOAL)
    
    graph = {
        'A': [('B', 1), ('C', 1), ('D', 5)],
        'B': [('A', 1), ('C', 1)],
        'C': [('A', 1), ('B', 1), ('GOAL', 2)],
        'D': [('A', 5), ('GOAL', 1)],
        'GOAL': [('C', 2), ('D', 1)]
    }

    # 2. Define node positions for the heuristic
    node_positions = {
        'A': [0, 2],
        'B': [1, 2],
        'C': [1, 1],
        'D': [0, 0],
        'GOAL': [2, 0]
    }
    
    # 3. Define the heuristic function that A* will use
    #    (It calls your euclidean_distance function)
    h = lambda node, goal: utils.euclidean_distance(
        node_positions[node], 
        node_positions[goal]
    )

    # 4. Run the search
    start_node = 'A'
    goal_node = 'GOAL'
    
    # Dijkstra's would go A->D->GOAL (Cost 6)
    # A* will be pulled by the heuristic to A->C->GOAL (Cost 1+2=3)
    
    path, cost = utils.aStarSearch(graph, start_node, goal_node, h)
    
    print(f"A* Path from {start_node} to {goal_node}:")
    print(f"  Path: {' -> '.join(path)}") # Expected: A -> C -> GOAL
    print(f"  Cost: {cost}") # Expected: 3
    
    print("\n--- All Tests Complete ---")