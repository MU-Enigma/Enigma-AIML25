import numpy as np
import time

class Perceptron:

    #Implements a single-layer Perceptron for binary classification
    #Hyperparameters are passed directly to the fit method

    def __init__(self):

        self.w = None
        self.b = None
        self.convergence_time = 0

    def _step_function(self, linear_output):

        #This is the Perceptron's activation function basically

        return np.where(linear_output >= 0, 1, 0)

    def fit(self, X, y, lr=0.01, epochs=1000):

        print("Starting Perceptron training...")
        start_time = time.time()
        
        # I. Init 
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        
        # II. Training Loop
        for i in range(epochs):
            for idx in range(n_samples):
                x_i = X[idx]
                y_i = y[idx]
                
                # --- Forward Pass (for one sample) ---
                linear_output = np.dot(x_i, self.w) + self.b
                y_pred = self._step_function(linear_output)
                
                # --- Perceptron Update Rule (Relaxation) ---
                if y_pred != y_i:
                    update = lr * (y_i - y_pred)
                    self.w += update * x_i
                    self.b += update

        self.convergence_time = time.time() - start_time
        print(f"Training finished in {self.convergence_time:.4f}s")

    def predict(self, X):

        if self.w is None or self.b is None:
            raise Exception("Model not trained yet. Hint: Call .fit() first")
            
        start_time = time.time()
        linear_output = np.dot(X, self.w) + self.b
        y_pred = self._step_function(linear_output)
        prediction_time = time.time() - start_time

        return y_pred, prediction_time