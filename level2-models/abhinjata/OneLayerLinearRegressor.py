import numpy as np
import time

class LinearRegression:

    #Implements Linear Regression using Gradient Descent
    
    #This is a single-layer model with no activation function, trained using Mean Squared Error (MSE) loss.

    def __init__(self, lr=0.01, epochs=1000):
        # Use variable names from your scripts
        self.lr = lr
        self.epochs = epochs
        self.w = None  # Replaces i_w, w_o
        self.b = None  # Replaces i_b, b_o
        self.convergence_time = 0

    def fit(self, X, y):

        print("Starting Linear Regression training...")
        start_time = time.time()
        
        # I. Initialization
        n_samples, n_features = X.shape
        
        # Single Layer Linear Regression has one set of weights and one bias
        self.w = np.zeros(n_features)
        self.b = 0
        
        # II. Training loop (Gradient Descent)
        for i in range(self.epochs):
            
            # Forward Press
            y_pred = np.dot(X, self.w) + self.b

            # MSE
            loss = np.mean((y_pred - y)**2)

            # Simple Backprop Gradiant Calc
            dy_pred = (2 / n_samples) * (y_pred - y)
            dw = np.dot(X.T, dy_pred)
            db = np.sum(dy_pred)

            # Fine Tune
            self.w -= self.lr * dw
            self.b -= self.lr * db
            
            if i % 100 == 0:
                print(f"  Epoch {i}, loss: {loss:.4f}")
        
        self.convergence_time = time.time() - start_time

        print(f"Training finished in {self.convergence_time:.4f}s")

    def predict(self, X):

        if self.w is None or self.b is None:
            raise Exception("Model not trained yet. Hint: call .fit() first")
            
        start_time = time.time()
        
        # Use the trained weights and bias
        y_pred = np.dot(X, self.w) + self.b
        
        prediction_time = time.time() - start_time
        return y_pred, prediction_time