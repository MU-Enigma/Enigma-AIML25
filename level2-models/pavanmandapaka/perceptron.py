import numpy as np


class Perceptron:
    """Perceptron """
    
    def __init__(self, learning_rate=0.01, max_iterations=1000):
        
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = None
        self.iterations_to_converge = 0
        self.error_history = []
    
    def fit(self, X, y):
       
        n_samples, n_features = X.shape
        
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.error_history = []
        
        for iteration in range(self.max_iterations):
            errors = 0
            
            for idx in range(n_samples):
                linear_output = np.dot(X[idx], self.weights) + self.bias
                y_pred = 1 if linear_output >= 0 else 0
                
                if y_pred != y[idx]:
                    update = self.learning_rate * (y[idx] - y_pred)
                    self.weights += update * X[idx]
                    self.bias += update
                    errors += 1
            
            self.error_history.append(errors)
            
            if errors == 0:
                self.iterations_to_converge = iteration + 1
                break
        else:
            self.iterations_to_converge = self.max_iterations
    
    def predict(self, X):
       
        linear_output = np.dot(X, self.weights) + self.bias
        return (linear_output >= 0).astype(int)
