import math

def l1_normalize(vector):
    l1_norm = sum(abs(x) for x in vector)

    if l1_norm == 0:
        return vector
    
    return [x / l1_norm for x in vector]

def l2_normalize(vector):
    l2_norm = math.sqrt(sum(x * x for x in vector))

    if l2_norm == 0:
        return vector
    
    return [x / l2_norm for x in vector]

def min_max_normalize(vector):
    if not vector:
        return vector
    
    min_val = min(vector)
    max_val = max(vector)

    if max_val == min_val:
        return [0.5] * len(vector)
    
    return [(x - min_val) / (max_val -min_val) for x in vector]

if __name__ == "__main__":
    print("=== Testing Normalization Functions ===")

    test_vec = [1, 2, 3, 4, 5]
    print(f"\nOriginal vector: {test_vec}")

    l1_result = l1_normalize(test_vec)
    print(f"\nL1 normalized: {[round(x,4) for x in l1_result]}")
    print(f"Sum of L1 normalized: {sum(l1_result):.4f}")

    l2_result = l2_normalize(test_vec)
    print(f"\nL2 normalized: {[round(x,4) for x in l2_result]}")
    print(f"Magnitude of L2 normalized: {math.sqrt(sum(x*x for x in l2_result)):.4f}")
    
    minmax_result = min_max_normalize(test_vec)
    print(f"\nMin-Max normalized: {minmax_result}")
    print(f"Range: [{min(minmax_result)}, {max(minmax_result)}]")

    test_vec2 = [-5, -2, 0, 2, 5]
    print(f"\n\nTesting with negative numbers: {test_vec2}")
    print(f"L1 normalized: {[round(x, 4) for x in l1_normalize(test_vec2)]}")
    print(f"L2 normalized: {[round(x, 4) for x in l2_normalize(test_vec2)]}")
    print(f"Min-Max normalized: {[round(x, 4) for x in min_max_normalize(test_vec2)]}")
    
