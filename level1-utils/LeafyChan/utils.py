import math

# Activation functions
def sigmoid(x: float) -> float:
    try:
        return 1 / (1 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

def relu(x: float) -> float:
    return max(0.0, x)

def tanh(x: float) -> float:
    return math.tanh(x)

# Vector operations
def dot_product(v1: list[float], v2: list[float]) -> float:
    if len(v1) != len(v2):
        raise ValueError("Vectors must have same length.")
    return sum(a * b for a, b in zip(v1, v2))

def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    num = dot_product(v1, v2)
    den = math.sqrt(dot_product(v1, v1)) * math.sqrt(dot_product(v2, v2))
    return num / den if den != 0 else 0.0

# Normalization methods
def normalize_l1(v: list[float]) -> list[float]:
    norm = sum(abs(x) for x in v)
    return [x / norm for x in v] if norm != 0 else v

def normalize_l2(v: list[float]) -> list[float]:
    norm = math.sqrt(sum(x ** 2 for x in v))
    return [x / norm for x in v] if norm != 0 else v

def normalize_minmax(v: list[float]) -> list[float]:
    v_min, v_max = min(v), max(v)
    if v_max == v_min:
        return [0.0 for _ in v]
    return [(x - v_min) / (v_max - v_min) for x in v]

# Simple heuristic:greedy choice (chooses the max value hehehehe)
def greedy_choice(numbers: list[int]) -> int:
    """Return the largest number (greedy heuristic)."""
    return max(numbers) if numbers else None

# ------------------- Tests -------------------
if __name__ == "__main__":
    print("Sigmoid(0):", sigmoid(0))
    print("ReLU(-3):", relu(-3))
    print("Tanh(1):", tanh(1))
    print("Dot Product([1,2,3],[4,5,6]):", dot_product([1,2,3],[4,5,6]))
    print("Cosine([1,0],[0,1]):", cosine_similarity([1,0],[0,1]))
    print("L1 Normalize([1,2,3]):", normalize_l1([1,2,3]))
    print("L2 Normalize([3,4]):", normalize_l2([3,4]))
    print("Min-Max Normalize([10,20,30]):", normalize_minmax([10,20,30]))
    print("Greedy Choice([4,1,9,2]):", greedy_choice([4,1,9,2]))
