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
    
    def sigmoid(self, x):
      import math
      return 1 / (1 + math.exp(-x))

    def predict_proba(self, X):
      # First get linear predictions
      linear_pred = self.predict(X)
      # Apply sigmoid to convert to probabilities
      return [self.sigmoid(p) for p in linear_pred]

    def predict_binary(self, X, threshold=0.5):
      probabilities = self.predict_proba(X)
      return [1 if p >= threshold else 0 for p in probabilities]

    def accuracy(self, X, y, threshold=0.5):
      predictions = self.predict_binary(X, threshold)
      correct = sum(1 for pred, true in zip(predictions, y) if pred == true)
      return correct / len(y)


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
    
    print("=== Linear Regression for Binary Classification ===\n")
    
    # Load data - convert to binary labels
    X, y_continuous = load_dataset('../../datasets/binary_classification.csv')
    
    # Convert continuous labels to binary (0 or 1)
    y = [1 if val > 0.5 else 0 for val in y_continuous]
    
    # Split into train/test (80/20)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Class distribution: {sum(y_train)} positive, {len(y_train) - sum(y_train)} negative\n")
    
    # Train model
    print("Training Linear Regression with MSE...")
    start_time = time.time()
    
    model = LinearRegression(learning_rate=0.0001, iterations=1000)
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    
    # Evaluate as classifier
    train_acc = model.accuracy(X_train, y_train)
    test_acc = model.accuracy(X_test, y_test)
    
    print(f"\n=== Classification Results ===")
    print(f"Training Accuracy: {train_acc * 100:.2f}%")
    print(f"Testing Accuracy: {test_acc * 100:.2f}%")
    print(f"Training Time: {training_time:.4f} seconds")
    
    # Also show R² for comparison
    train_r2 = model.score(X_train, y_train)
    test_r2 = model.score(X_test, y_test)
    print(f"\nR² Score (regression metric):")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Testing R²: {test_r2:.4f}")
    
    # Sample predictions
    print(f"\n=== Sample Predictions ===")
    sample_X = X_test[:10]
    sample_y_true = y_test[:10]
    sample_y_pred = model.predict_binary(sample_X)
    sample_y_proba = model.predict_proba(sample_X)
    
    for i in range(10):
        status = "[CORRECT]" if sample_y_pred[i] == sample_y_true[i] else "[WRONG]"
        print(f"{status} Input: {sample_X[i]:.2f} | True: {sample_y_true[i]} | Predicted: {sample_y_pred[i]} | Probability: {sample_y_proba[i]:.4f}")