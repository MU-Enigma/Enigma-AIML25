import math

def sigmoid(x):
    return 1/ (1+ math.exp(-x))

def relu(x):
    return max(0, x)

def tanh(x):
    return math.tanh(x)

if __name__ == "__main__":
    print("=== Testing Activation Functions ===")
    test_values = [-2, -1, 0, 1, 2]

    for val in test_values:
        print(f"\nInput: {val}")
        print(f" Sigmoid: {sigmoid(val):.4f}")
        print(f" ReLU: {relu(val):.4f}")
        print(f" Tanh: {tanh(val):.4f}")
        