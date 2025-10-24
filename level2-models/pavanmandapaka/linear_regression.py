import numpy as np


class LinearRegression:
    """
    Linear Regression with Gradient Descent for binary classification.
    Uses a threshold of 0.5 to convert continuous predictions to binary labels.
    """
    
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        """
        Initialize Linear Regression model.
        
        Args:
            learning_rate: Step size for gradient descent
            max_iterations: Maximum number of training iterations
            tolerance: Convergence threshold for gradient magnitude
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        self.iterations_to_converge = 0
    
    def fit(self, X, y):
        """
        Train the linear regression model using gradient descent.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
        """
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for iteration in range(self.max_iterations):
            # Forward pass: compute predictions
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Check for convergence
            if np.linalg.norm(dw) < self.tolerance:
                self.iterations_to_converge = iteration + 1
                break
        else:
            # Max iterations reached without convergence
            self.iterations_to_converge = self.max_iterations
    
    def predict(self, X):
        """
        Make binary predictions using a threshold of 0.5.
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            Binary predictions (n_samples,)
        """
        linear_output = np.dot(X, self.weights) + self.bias
        return (linear_output >= 0.5).astype(int)
