import math

def dot_product(vec1, vec2):
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have the same length")
    return sum(a * b for a, b in zip(vec1, vec2))

def magnitude(vector):
    return math.sqrt(sum(x * x for x in vector))

def cosine_similarity(vec1,vec2):
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have the same length")
    
    dot_prod = dot_product(vec1, vec2)

    mag1 = magnitude(vec1)
    mag2 = magnitude(vec2)

    if mag1 == 0 or mag2 == 0:
        return 0
    
    return dot_prod / (mag1 * mag2)

if __name__ == "__main__":
    print("=== Testing Vector Operations ===")

    v1 = [1, 2, 3]
    v2 = [4, 5, 6]
    v3 = [1, 2, 3]
    v4 = [-1, -2, -3]

    print(f"\nVector 1: {v1}")
    print(f"Vector 2: {v2}")
    print(f"Dot product: {dot_product(v1, v2)}")
    print(f"Cosine similarity: {cosine_similarity(v1, v2):.4f}")

    print(f"\nVector 1: {v1}")
    print(f"Vector 3 (identical): {v3}")
    print(f"Cosine similarity: {cosine_similarity(v1, v3):.4f}")

    print(f"\nVector 1: {v1}")
    print(f"Vector 4 (opposite): {v4}")
    print(f"Cosine similarity: {cosine_similarity(v1, v4):.4f}")
    
