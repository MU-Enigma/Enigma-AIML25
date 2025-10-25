import numpy as np
import time

class LinearRegressionGD:
    def __init__(self, lr=0.01, epochs=1000, tolerance=1e-6):
        self.lr = lr
        self.epochs = epochs
        self.tolerance = tolerance
        self.weights = None
        self.bias = 0
        self.training_time = 0

    def fit(self, X, y):
        start_time = time.time()
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        prev_loss = float("inf")
        for _ in range(self.epochs):
            y_pred = np.dot(X, self.weights) + self.bias
            errors = y_pred - y

            # Gradients
            dw = (1 / n_samples) * np.dot(X.T, errors)
            db = (1 / n_samples) * np.sum(errors)

            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # Mean Squared Error
            loss = np.mean(errors ** 2)

            # Check convergence
            if abs(prev_loss - loss) < self.tolerance:
                break
            prev_loss = loss

        self.training_time = time.time() - start_time

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
