import math
from functions import *

def run_tests():
    """Runs a series of tests for all implemented utility functions."""
    
    print("--- Running Level 1 Utility Function Tests ---")

    # --- Test Activation Functions ---
    print("\n[Activation Functions]")
    print(f"sigmoid(1.0)  = {sigmoid(1.0):.4f}  (Expected: ~0.7311)")
    print(f"sigmoid(-2.0) = {sigmoid(-2.0):.4f} (Expected: ~0.1192)")
    print(f"sigmoid(0.0)  = {sigmoid(0.0):.4f}  (Expected: 0.5)")
    
    print(f"relu(5.5)     = {relu(5.5)}    (Expected: 5.5)")
    print(f"relu(-5.5)    = {relu(-5.5)}   (Expected: 0.0)")
    
    print(f"tanh(1.0)     = {tanh(1.0):.4f}   (Expected: ~0.7616)")
    print(f"tanh(-1.0)    = {tanh(-1.0):.4f}  (Expected: ~-0.7616)")

    # --- Test Vector/Similarity Functions ---
    print("\n[Vector/Similarity Functions]")
    v1 = [1, 2, 3]
    v2 = [4, 5, 6]
    v_ortho_1 = [1, 0]
    v_ortho_2 = [0, 1]
    
    print(f"dot_product({v1}, {v2}) = {dot_product(v1, v2)} (Expected: 32)")
    
    cos_sim = cosine_similarity(v1, v2)
    print(f"cosine_similarity({v1}, {v2}) = {cos_sim:.4f} (Expected: ~0.9746)")
    
    cos_sim_ortho = cosine_similarity(v_ortho_1, v_ortho_2)
    print(f"cosine_similarity({v_ortho_1}, {v_ortho_2}) = {cos_sim_ortho:.4f} (Expected: 0.0)")

    # --- Test Normalization Functions ---
    print("\n[Normalization Functions]")
    vec_norm = [1, -2, 3]
    data_scale = [10, 20, 30, 40, 50]
    
    l1 = l1_normalize(vec_norm)
    print(f"l1_normalize({vec_norm}) = [{', '.join(f'{x:.2f}' for x in l1)}] (Sum abs: {sum(abs(x) for x in l1):.1f})")
    
    l2 = l2_normalize(vec_norm)
    print(f"l2_normalize({vec_norm}) = [{', '.join(f'{x:.2f}' for x in l2)}] (Magnitude: {math.sqrt(sum(x**2 for x in l2)):.1f})")
    
    min_max = min_max_normalize(data_scale)
    print(f"min_max_normalize({data_scale}) = {min_max}")

    # --- Test Heuristic Function ---
    print("\n[Heuristic Function]")
    
    # 1. Test for winning move (Player 'X' can win at index 2)
    board_win = ['X', 'X', ' ', 'O', 'O', ' ', ' ', ' ', ' ']
    move_win = find_greedy_tic_tac_toe_move(board_win, 'X')
    print(f"Test 1 (Win): Move = {move_win} (Expected: 2)")

    # 2. Test for blocking move (Player 'X' must block 'O' at index 2)
    board_block = ['O', 'O', ' ', 'X', ' ', ' ', 'X', ' ', ' ']
    move_block = find_greedy_tic_tac_toe_move(board_block, 'X')
    print(f"Test 2 (Block): Move = {move_block} (Expected: 2)")

    # 3. Test for center (No win/block, 'X' should take center)
    board_center = ['X', ' ', ' ', ' ', ' ', ' ', 'O', ' ', ' ']
    move_center = find_greedy_tic_tac_toe_move(board_center, 'X')
    print(f"Test 3 (Center): Move = {move_center} (Expected: 4)")

    # 4. Test for corner (Center taken, 'O' should take a corner)
    board_corner = [' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ']
    move_corner = find_greedy_tic_tac_toe_move(board_corner, 'O')
    print(f"Test 4 (Corner): Move = {move_corner} (Expected: 0, 2, 6, or 8)")

    print("\n--- Tests Complete! ---")
    print("If all 'Expected' values match the results, you are ready to commit.")

if __name__ == "__main__":
    run_tests()