import math
import random

class LinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.slope = 0  # m in y = mx + b
        self.intercept = 0  # b in y = mx + b
        self.losses = []  # Track error over time
    
    def predict(self, X):
        return [self.slope * x + self.intercept for x in X]
    
    def fit(self, X, y):
        n = len(X)
        
        for iteration in range(self.iterations):
            # Make predictions with current parameters
            y_pred = self.predict(X)
            
            # Calculate error (Mean Squared Error)
            mse = sum((y_pred[i] - y[i]) ** 2 for i in range(n)) / n
            self.losses.append(mse)
            
            # Calculate gradients (derivatives)
            # How much should we adjust slope and intercept?
            d_slope = sum(2 * (y_pred[i] - y[i]) * X[i] for i in range(n)) / n
            d_intercept = sum(2 * (y_pred[i] - y[i]) for i in range(n)) / n
            
            # Update parameters (move in opposite direction of gradient)
            self.slope -= self.learning_rate * d_slope
            self.intercept -= self.learning_rate * d_intercept
            
            # Print progress every 100 iterations
            if (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}: Loss = {mse:.4f}")
        
        print(f"\nFinal parameters: slope = {self.slope:.4f}, intercept = {self.intercept:.4f}")
    
    def score(self, X, y):
        y_pred = self.predict(X)
        y_mean = sum(y) / len(y)
        
        # Total sum of squares
        ss_tot = sum((y[i] - y_mean) ** 2 for i in range(len(y)))
        
        # Residual sum of squares
        ss_res = sum((y[i] - y_pred[i]) ** 2 for i in range(len(y)))
        
        r2 = 1 - (ss_res / ss_tot)
        return r2


def load_dataset(filename):
    X = []
    y = []
    
    try:
        with open(filename, 'r') as f:
            # Skip header
            next(f)
            
            for line in f:
                values = line.strip().split(',')
                # Assuming last column is target, rest are features
                # For simplicity, we'll use just the first feature
                X.append(float(values[0]))
                y.append(float(values[-1]))
        
        print(f"Loaded {len(X)} samples from {filename}")
        return X, y
    
    except FileNotFoundError:
        print(f"Error: {filename} not found. Using synthetic data for demo.")
        # Generate synthetic data for testing
        X = [i for i in range(100)]
        y = [2 * x + 5 + random.uniform(-10, 10) for x in X]
        return X, y


# Test the model
if __name__ == "__main__":
    import time
    
    print("=== Linear Regression Implementation ===\n")
    
    # Load data
    X, y = load_dataset('../../datasets/binary_classification.csv')
    
    # Split into train/test (80/20)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}\n")
    
    # Train model
    print("Training Linear Regression...")
    start_time = time.time()
    
    model = LinearRegression(learning_rate=0.0001, iterations=1000)
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"\n=== Results ===")
    print(f"Training R² Score: {train_score:.4f}")
    print(f"Testing R² Score: {test_score:.4f}")
    print(f"Training Time: {training_time:.4f} seconds")
    
    # Make sample predictions
    print(f"\n=== Sample Predictions ===")
    sample_X = X_test[:5]
    sample_y_true = y_test[:5]
    sample_y_pred = model.predict(sample_X)
    
    for i in range(5):
        print(f"Input: {sample_X[i]:.2f} | True: {sample_y_true[i]:.2f} | Predicted: {sample_y_pred[i]:.2f}")