import random
import time

class Perceptron:
    def __init__(self, learning_rate=0.01, iterations=1000):
        """
        Initialize the perceptron
        """
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = 0
        self.errors = []  # Track errors over time
    
    def activation(self, z):
        return 1 if z >= 0 else 0
    
    def predict_single(self, x):
        # Handle single feature (convert float to list)
        if not isinstance(x, list):
          x = [x]
    
        z = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
        return self.activation(z)
    
    def predict(self, X):
        return [self.predict_single(x) for x in X]
    
    def fit(self, X, y):
        n_samples = len(X)
        n_features = len(X[0]) if isinstance(X[0], list) else 1
        
        # Initialize weights randomly
        self.weights = [random.uniform(-1, 1) for _ in range(n_features)]
        self.bias = random.uniform(-1, 1)
        
        print("Training Perceptron...")
        
        for iteration in range(self.iterations):
            errors = 0
            
            # Go through each training sample
            for i in range(n_samples):
                xi = X[i] if isinstance(X[i], list) else [X[i]]
                yi = y[i]
                
                # Make prediction
                prediction = self.predict_single(xi)
                
                # Calculate error
                error = yi - prediction
                
                if error != 0:
                    errors += 1
                    # Update weights: w = w + learning_rate * error * x
                    for j in range(len(self.weights)):
                        self.weights[j] += self.learning_rate * error * xi[j]
                    self.bias += self.learning_rate * error
            
            self.errors.append(errors)
            
            # Print progress
            if (iteration + 1) % 100 == 0:
                accuracy = (n_samples - errors) / n_samples * 100
                print(f"Iteration {iteration + 1}: Errors = {errors}, Accuracy = {accuracy:.2f}%")
            
            # Early stopping if perfect classification
            if errors == 0:
                print(f"Converged at iteration {iteration + 1}!")
                break
    
    def accuracy(self, X, y):
        predictions = self.predict(X)
        correct = sum(1 for pred, true in zip(predictions, y) if pred == true)
        return correct / len(y)


def load_binary_dataset(filename):
    X = []
    y = []
    
    try:
        with open(filename, 'r') as f:
            # Skip header
            next(f)
            
            for line in f:
                values = line.strip().split(',')
                # Use first feature and convert label to binary
                X.append(float(values[0]))
                # Convert to binary (0 or 1)
                y.append(1 if float(values[-1]) > 0.5 else 0)
        
        print(f"Loaded {len(X)} samples from {filename}")
        return X, y
    
    except FileNotFoundError:
        print(f"Error: {filename} not found. Using synthetic data.")
        # Generate linearly separable data
        X = [random.uniform(0, 10) for _ in range(100)]
        y = [1 if x > 5 else 0 for x in X]
        return X, y


# Test the model
if __name__ == "__main__":
    print("=== Perceptron Implementation ===\n")
    
    # Load data
    X, y = load_binary_dataset('../../datasets/binary_classification.csv')
    
    # Split into train/test (80/20)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Class distribution: {sum(y_train)} positive, {len(y_train) - sum(y_train)} negative\n")
    
    # Train model
    start_time = time.time()
    
    model = Perceptron(learning_rate=0.1, iterations=1000)
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    
    # Evaluate
    train_acc = model.accuracy(X_train, y_train)
    test_acc = model.accuracy(X_test, y_test)
    
    print(f"\n=== Results ===")
    print(f"Training Accuracy: {train_acc * 100:.2f}%")
    print(f"Testing Accuracy: {test_acc * 100:.2f}%")
    print(f"Training Time: {training_time:.4f} seconds")
    
    # Sample predictions
    print(f"\n=== Sample Predictions ===")
    sample_X = X_test[:10]
    sample_y_true = y_test[:10]
    sample_y_pred = model.predict(sample_X)
    
    for i in range(10):
        status = "[CORRECT]" if sample_y_pred[i] == sample_y_true[i] else "[WRONG]"
        print(f"{status} Input: {sample_X[i]:.2f} | True: {sample_y_true[i]} | Predicted: {sample_y_pred[i]}")