import math

def sigmoid(x):
    return 1/(1 + math.exp(-x))

def ReLu(x):
    return max(x, 0)


def tanh(x):
    num = math.exp(x) - math.exp(-x)
    den = math.exp(x) + math.exp(-x)
    return num/den

def vector_dot_product(vector_a, vector_b):
  # Ensure the vectors have the same dimension
  if len(vector_a) != len(vector_b):
    raise ValueError("Vectors must have the same length to calculate the dot product.")

  # Calculate the sum of the products of corresponding elements
  dot_product = sum(a * b for a, b in zip(vector_a, vector_b))

  return dot_product

def _l2(x):
    """L2 norm."""
    s = 0.0
    for v in x:
        fv = float(v)
        if not math.isfinite(fv):
            raise ValueError("L2: non-finite value")
        s += fv * fv
    return math.sqrt(s)


def cosine_similarity(a, b):
    
    if len(a) != len(b):
        raise ValueError(f"cosine: length mismatch {len(a)} != {len(b)}")
    
    na, nb = _l2(a), _l2(b)
    
    if na == 0.0 or nb == 0.0:
        raise ValueError("cosine: zero-norm vector")
        
    return vector_dot_product(a, b) / (na * nb)

import math

def _l2(x):
    s = 0.0
    for v in x:
        fv = float(v)
        if not math.isfinite(fv):
            raise ValueError("L2: non-finite value")
        s += fv * fv
    return math.sqrt(s)


def normalize_l1(x):
    s = sum(abs(float(v)) for v in x)
    if s == 0.0:
        raise ValueError("L1: sum(|x|)=0")
    return [float(v) / s for v in x]


def normalize_l2(x):
    n = _l2(x)
    if n == 0.0:
        raise ValueError("L2: ||x||=0")
    return [float(v) / n for v in x]


def normalize_minmax(x, low=0.0, high=1.0):
    if not (low < high):
        raise ValueError("minmax: low < high required")
        
    xs = [float(v) for v in x]
    mn, mx = min(xs), max(xs)
    
    if mx == mn:
        return [low] * len(xs) 
        
    scale = (high - low) / (mx - mn)
    
    return [low + (v - mn) * scale for v in xs]
