import numpy as np


class LinearRegression:
    """ Linear Regression with Gradient Descent """
    
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        self.iterations_to_converge = 0
        self.loss_history = []
    
    def fit(self, X, y):
        
        n_samples, n_features = X.shape
        
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.loss_history = []
        
        for iteration in range(self.max_iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # MSE loss
            mse_loss = np.mean((y_pred - y) ** 2)
            self.loss_history.append(mse_loss)
            
            # convergence check
            grad_norm = np.linalg.norm(dw)
            if grad_norm < self.tolerance:
                self.iterations_to_converge = iteration + 1
                break
        else:
            self.iterations_to_converge = self.max_iterations
    
    def predict(self, X):
       
        linear_output = np.dot(X, self.weights) + self.bias
        return (linear_output >= 0.5).astype(int)
