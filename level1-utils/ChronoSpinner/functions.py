import math
from typing import List

def sigmoid(x: float) -> float:
    """
    Compute the sigmoid activation function for a given input.
    Args:
        x (float): input value
    Returns:
        float: sigmoid of input
    """
    return 1 / (1 + math.exp(-x))

def relu(x: float) -> float:
    """
    Compute the ReLU activation function for a given input.
    Args:
        x (float): input value
    Returns:
        float: result of ReLU(x)
    """
    return max(0.0, x)

def tanh(x: float) -> float:
    """
    Compute the tanh activation function for a given input.
    Args:
        x (float): input value
    Returns:
        float: tanh of input
    """
    return math.tanh(x)

def dot_product(v1: List[float], v2: List[float]) -> float:
    """
    Compute the dot product of two vectors.
    Args:
        v1, v2 (List[float]): input vectors (equal length)
    Returns:
        float: dot product
    """
    return sum(a * b for a, b in zip(v1, v2))

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """
    Compute the cosine similarity between two vectors.
    Args:
        v1, v2 (List[float]): input vectors (equal length)
    Returns:
        float: cosine similarity
    """
    dot = dot_product(v1, v2)
    norm1 = math.sqrt(dot_product(v1, v1))
    norm2 = math.sqrt(dot_product(v2, v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

def l1_normalize(v: List[float]) -> List[float]:
    """
    Normalize a vector using L1 norm.
    Args:
        v (List[float]): input vector
    Returns:
        List[float]: L1-normalized vector
    """
    norm = sum(abs(x) for x in v)
    if norm == 0:
        return v[:]
    return [x / norm for x in v]

def l2_normalize(v: List[float]) -> List[float]:
    """
    Normalize a vector using L2 norm.
    Args:
        v (List[float]): input vector
    Returns:
        List[float]: L2-normalized vector
    """
    norm = math.sqrt(sum(x*x for x in v))
    if norm == 0:
        return v[:]
    return [x / norm for x in v]

def min_max_normalize(v: List[float]) -> List[float]:
    """
    Normalize a vector using min-max scaling to [0, 1].
    Args:
        v (List[float]): input vector
    Returns:
        List[float]: min-max normalized vector
    """
    min_v = min(v)
    max_v = max(v)
    if max_v == min_v:
        return [0.0 for _ in v]
    return [(x - min_v) / (max_v - min_v) for x in v]

def heuristic_tic_tac_toe(board: List[str], player: str) -> int:
    """
    Simple heuristic for Tic-Tac-Toe: returns score based on number of possible winning lines for the player.
    Args:
        board (List[str]): 1D list of length 9 representing board
        player (str): 'X' or 'O'
    Returns:
        int: score based on immediate winning moves available
    """
    wins = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
    score = 0
    for combo in wins:
        line = [board[i] for i in combo]
        if line.count(player) == 2 and line.count(' ') == 1:
            score += 1
    return score

def main():
    x = 0.5
    print(f"sigmoid({x}) = {sigmoid(x)}")
    print(f"ReLU({x}) = {relu(x)}")
    print(f"tanh({x}) = {tanh(x)}")

    v1 = [1, 2, 3]
    v2 = [4, 5, 6]

    print(f"dot_product({v1}, {v2}) = {dot_product(v1, v2)}")
    print(f"cosine_similarity({v1}, {v2}) = {cosine_similarity(v1, v2)}")

    print(f"L1 normalize {v1} = {l1_normalize(v1)}")
    print(f"L2 normalize {v1} = {l2_normalize(v1)}")
    print(f"Min-Max normalize {v1} = {min_max_normalize(v1)}")

    board = ['X', 'O', 'X',
             ' ', 'X', ' ',
             'O', ' ', 'O']
    player = 'X'
    print(f"Heuristic score for player '{player}' on board {board} = {heuristic_tic_tac_toe(board, player)}")

if __name__ == "__main__":
    main()
