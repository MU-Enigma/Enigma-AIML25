import numpy as np
import time

class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, max_iters=1000, tol=1e-6):
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.tol = tol
        self.w = None
        self.b = 0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        start_time = time.time()
        for i in range(self.max_iters):
            y_pred = np.dot(X, self.w) + self.b
            error = y_pred - y
            grad_w = (1/n_samples) * np.dot(X.T, error)
            grad_b = (1/n_samples) * np.sum(error)

            # Update weights
            self.w -= self.learning_rate * grad_w
            self.b -= self.learning_rate * grad_b

            # Check for convergence
            if np.linalg.norm(grad_w) < self.tol and abs(grad_b) < self.tol:
                break
        self.time_to_converge = time.time() - start_time

    def predict(self, X):
        start_time = time.time()
        y_pred = np.dot(X, self.w) + self.b
        self.prediction_time = (time.time() - start_time) / X.shape[0]
        # For classification, threshold at 0.5
        return (y_pred >= 0.5).astype(int)


class Perceptron:
    def __init__(self, learning_rate=0.01, max_iters=1000):
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.w = None
        self.b = 0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        start_time = time.time()
        for _ in range(self.max_iters):
            errors = 0
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.w) + self.b
                y_pred = 1 if linear_output >= 0 else 0
                update = self.learning_rate * (y[idx] - y_pred)
                self.w += update * x_i
                self.b += update
                errors += int(update != 0.0)
            if errors == 0:  # Converged
                break
        self.time_to_converge = time.time() - start_time

    def predict(self, X):
        start_time = time.time()
        linear_output = np.dot(X, self.w) + self.b
        y_pred = (linear_output >= 0).astype(int)
        self.prediction_time = (time.time() - start_time) / X.shape[0]
        return y_pred

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json

def load_data(filepath):
    data = pd.read_csv(filepath)
    X = data[['feature1', 'feature2']].values
    y = data['label'].values
    return X, y

def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    metrics = {
        'accuracy': accuracy,
        'time_to_convergence': model.time_to_converge,
        'time_per_prediction': model.prediction_time
    }
    return metrics

def main(filepath):
    # Load dataset
    X, y = load_data(filepath)
    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize models
    lr_model = LinearRegressionGD()
    perceptron_model = Perceptron()

    # Evaluate Linear Regression
    lr_metrics = evaluate_model(lr_model, X_train, y_train, X_test, y_test)
    print("Linear Regression Metrics:", lr_metrics)

    # Evaluate Perceptron
    perceptron_metrics = evaluate_model(perceptron_model, X_train, y_train, X_test, y_test)
    print("Perceptron Metrics:", perceptron_metrics)

    # Save metrics to JSON
    all_metrics = {
        "LinearRegression": lr_metrics,
        "Perceptron": perceptron_metrics
    }
    with open('metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=4)

if __name__ == "__main__":
    main('binary_classification.csv')
