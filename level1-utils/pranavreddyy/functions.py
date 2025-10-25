import math
from typing import List, Union

NumericVector = List[Union[int, float]]


# --- 1. Activation Functions ---

def sigmoid(x: float) -> float:
    """
    Calculates the sigmoid activation function.
    
    Args:
        x: A numeric value.
    
    Returns:
        The sigmoid of x, a value between 0 and 1.
    """
    # We use this check to avoid OverflowError: math.exp(x)
    # for large negative x (e.g., x < -709)
    if x < -709:
        return 0.0
    return 1 / (1 + math.exp(-x))

def relu(x: float) -> float:
    """
    Calculates the Rectified Linear Unit (ReLU) activation function.
    
    Args:
        x: A numeric value.
    
    Returns:
        x if x > 0, else 0.
    """
    return max(0.0, x)

def tanh(x: float) -> float:
    """
    Calculates the hyperbolic tangent (tanh) activation function.
    
    Args:
        x: A numeric value.
    
    Returns:
        The tanh of x, a value between -1 and 1.
    """
    return math.tanh(x)


# --- 2. Vector Dot Product and Cosine Similarity ---

def dot_product(v1: NumericVector, v2: NumericVector) -> float:
    """
    Calculates the dot product of two vectors.
    
    Args:
        v1: The first vector.
        v2: The second vector.
    
    Returns:
        The dot product.
        
    Raises:
        ValueError: If the vectors are not of the same length.
    """
    if len(v1) != len(v2):
        raise ValueError("Vectors must be of the same length.")
    return sum(x * y for x, y in zip(v1, v2))

def cosine_similarity(v1: NumericVector, v2: NumericVector) -> float:
    """
    Calculates the cosine similarity between two non-zero vectors.
    
    Args:
        v1: The first vector.
        v2: The second vector.
    
    Returns:
        The cosine similarity, a value between -1 and 1.
    """
    prod = dot_product(v1, v2)
    
    # Calculate magnitudes (L2 norm)
    mag_v1 = math.sqrt(dot_product(v1, v1))
    mag_v2 = math.sqrt(dot_product(v2, v2))
    
    if mag_v1 == 0 or mag_v2 == 0:
        # Handle the case of a zero vector to avoid division by zero
        return 0.0
        
    return prod / (mag_v1 * mag_v2)


# --- 3. Normalization Functions ---

def l1_normalize(v: NumericVector) -> List[float]:
    """
    Normalizes a vector using L1 Normalization (Manhattan norm).
    Each element is divided by the sum of the absolute values of all elements.
    
    Args:
        v: The vector to normalize.
    
    Returns:
        A new list with the L1-normalized values.
    """
    norm = sum(abs(x) for x in v)
    if norm == 0:
        # Avoid division by zero if it's a zero vector
        return [0.0] * len(v)
    return [x / norm for x in v]

def l2_normalize(v: NumericVector) -> List[float]:
    """
    Normalizes a vector using L2 Normalization (Euclidean norm).
    Each element is divided by the magnitude (square root of the sum of squares).
    
    Args:
        v: The vector to normalize.
    
    Returns:
        A new list with the L2-normalized values (unit vector).
    """
    norm = math.sqrt(sum(x**2 for x in v))
    if norm == 0:
        # Avoid division by zero if it's a zero vector
        return [0.0] * len(v)
    return [x / norm for x in v]

def min_max_normalize(data: NumericVector) -> List[float]:
    """
    Normalizes a vector using Min-Max scaling to the range [0, 1].
    
    Args:
        data: The vector of data to normalize.
    
    Returns:
        A new list with values scaled between 0 and 1.
    """
    if not data:
        return []
        
    min_val = min(data)
    max_val = max(data)
    range_val = max_val - min_val
    
    if range_val == 0:
        # All elements are the same; return a list of 0s
        return [0.0] * len(data)
        
    return [(x - min_val) / range_val for x in data]


# --- 4. Heuristic Function ---

def find_greedy_tic_tac_toe_move(board: List[str], player: str) -> int:
    """
    Finds a greedy move for a Tic-Tac-Toe game based on a simple heuristic.
    
    Board is represented as a list of 9 strings ('X', 'O', or ' ' for empty).
    
    Heuristic Priority:
    1. Win: Find a move that wins the game.
    2. Block: Find a move that blocks an opponent's win.
    3. Center: Take the center square.
    4. Corner: Take an empty corner.
    5. Side: Take an empty side.

    Args:
        board: The 9-element list representing the game board (indices 0-8).
        player: The current player's marker ('X' or 'O').
    
    Returns:
        An integer (0-8) for the best move, or -1 if no moves are possible.
    """
    
    opponent = 'O' if player == 'X' else 'X'
    
    # Helper to check for a win on a given board state for a given player
    def check_win(b: List[str], p: str) -> bool:
        win_conditions = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8), # Rows
            (0, 3, 6), (1, 4, 7), (2, 5, 8), # Cols
            (0, 4, 8), (2, 4, 6)             # Diagonals
        ]
        for wc in win_conditions:
            if b[wc[0]] == p and b[wc[1]] == p and b[wc[2]] == p:
                return True
        return False

    # Get all available (empty) moves
    available_moves = [i for i, spot in enumerate(board) if spot == ' ']
    if not available_moves:
        return -1 # Board is full

    # --- 1. Check for a winning move ---
    for move in available_moves:
        # Try the move
        board[move] = player
        if check_win(board, player):
            board[move] = ' ' # Reset board
            return move
        board[move] = ' ' # Reset board

    # --- 2. Check to block opponent's winning move ---
    for move in available_moves:
        # Try the move for the opponent
        board[move] = opponent
        if check_win(board, opponent):
            board[move] = ' ' # Reset board
            return move
        board[move] = ' ' # Reset board
        
    # --- 3. Take the center ---
    if 4 in available_moves:
        return 4
        
    # --- 4. Take an empty corner ---
    corners = [0, 2, 6, 8]
    for move in available_moves:
        if move in corners:
            return move
            
    # --- 5. Take an empty side ---
    # This will be the only remaining option if corners/center are gone
    return available_moves[0]