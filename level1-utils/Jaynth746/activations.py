import math

def sigmoid(x):
    """
    Sigmoid activation function: σ(x) = 1 / (1 + e^(-x))
    
    Args:
        x: Input value (float or int)
    
    Returns:
        Sigmoid of x
    """
    return 1 / (1 + math.exp(-x))

def relu(x):
    """
    ReLU (Rectified Linear Unit) activation function: max(0, x)
    
    Args:
        x: Input value (float or int)
    
    Returns:
        ReLU of x
    """
    return max(0, x)

def tanh(x):
    """
    Hyperbolic tangent activation function
    
    Args:
        x: Input value (float or int)
    
    Returns:
        Tanh of x
    """
    return math.tanh(x)

# Test the functions
if __name__ == "__main__":
    test_value = 0.5
    print(f"sigmoid({test_value}) = {sigmoid(test_value)}")
    print(f"relu({test_value}) = {relu(test_value)}")
    print(f"tanh({test_value}) = {tanh(test_value)}")
    
    test_value = -1.0
    print(f"\nsigmoid({test_value}) = {sigmoid(test_value)}")
    print(f"relu({test_value}) = {relu(test_value)}")
    print(f"tanh({test_value}) = {tanh(test_value)}")
