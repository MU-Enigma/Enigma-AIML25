import numpy as np


class Perceptron:
    """
    Perceptron algorithm for binary classification.
    A simple linear classifier that updates weights based on misclassified samples.
    """
    
    def __init__(self, learning_rate=0.01, max_iterations=1000):
        """
        Initialize Perceptron model.
        
        Args:
            learning_rate: Step size for weight updates
            max_iterations: Maximum number of training epochs
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = None
        self.iterations_to_converge = 0
    
    def fit(self, X, y):
        """
        Train the perceptron using the perceptron learning rule.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
        """
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Training loop
        for iteration in range(self.max_iterations):
            errors = 0
            
            # Iterate through each training sample
            for idx in range(n_samples):
                # Compute linear output
                linear_output = np.dot(X[idx], self.weights) + self.bias
                
                # Apply activation (step function)
                y_pred = 1 if linear_output >= 0 else 0
                
                # Update weights if prediction is incorrect
                if y_pred != y[idx]:
                    update = self.learning_rate * (y[idx] - y_pred)
                    self.weights += update * X[idx]
                    self.bias += update
                    errors += 1
            
            # Check for convergence (no errors in this epoch)
            if errors == 0:
                self.iterations_to_converge = iteration + 1
                break
        else:
            # Max iterations reached without convergence
            self.iterations_to_converge = self.max_iterations
    
    def predict(self, X):
        """
        Make binary predictions.
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            Binary predictions (n_samples,)
        """
        linear_output = np.dot(X, self.weights) + self.bias
        return (linear_output >= 0).astype(int)
