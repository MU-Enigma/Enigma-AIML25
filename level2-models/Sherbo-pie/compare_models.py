import time
from linear_regression import LinearRegression, load_dataset
from perceptron import Perceptron, load_binary_dataset

def compare_models():
    """
    Compare Linear Regression and Perceptron on the dataset
    Both evaluated as classifiers using accuracy
    """
    print("=" * 60)
    print("MODEL COMPARISON: Linear Regression vs Perceptron")
    print("(Both evaluated as binary classifiers)")
    print("=" * 60)
    
    # Load datasets
    print("\n[1] Loading datasets...")
    X_reg, y_reg_continuous = load_dataset('../../datasets/binary_classification.csv')
    X_perc, y_perc = load_binary_dataset('../../datasets/binary_classification.csv')
    
    # Convert Linear Regression targets to binary
    y_reg = [1 if val > 0.5 else 0 for val in y_reg_continuous]
    
    # Split data (80/20)
    split_idx = int(0.8 * len(X_reg))
    
    X_train_reg, X_test_reg = X_reg[:split_idx], X_reg[split_idx:]
    y_train_reg, y_test_reg = y_reg[:split_idx], y_reg[split_idx:]
    
    X_train_perc, X_test_perc = X_perc[:split_idx], X_perc[split_idx:]
    y_train_perc, y_test_perc = y_perc[:split_idx], y_perc[split_idx:]
    
    print(f"Dataset size: {len(X_reg)} samples")
    print(f"Training: {len(X_train_reg)} | Testing: {len(X_test_reg)}")
    print(f"Class distribution: {sum(y_train_reg)} positive, {len(y_train_reg) - sum(y_train_reg)} negative")
    
    # ========== LINEAR REGRESSION AS CLASSIFIER ==========
    print("\n" + "=" * 60)
    print("[2] Training Linear Regression (with sigmoid classification)...")
    print("=" * 60)
    
    lr_start = time.time()
    lr_model = LinearRegression(learning_rate=0.0001, iterations=1000)
    lr_model.fit(X_train_reg, y_train_reg)
    lr_training_time = time.time() - lr_start
    
    # Classification accuracy
    lr_train_acc = lr_model.accuracy(X_train_reg, y_train_reg)
    lr_test_acc = lr_model.accuracy(X_test_reg, y_test_reg)
    
    # Time per prediction
    pred_start = time.time()
    _ = lr_model.predict_binary(X_test_reg)
    lr_pred_time = (time.time() - pred_start) / len(X_test_reg)
    
    # ========== PERCEPTRON ==========
    print("\n" + "=" * 60)
    print("[3] Training Perceptron...")
    print("=" * 60)
    
    perc_start = time.time()
    perc_model = Perceptron(learning_rate=0.1, iterations=1000)
    perc_model.fit(X_train_perc, y_train_perc)
    perc_training_time = time.time() - perc_start
    
    # Classification accuracy
    perc_train_acc = perc_model.accuracy(X_train_perc, y_train_perc)
    perc_test_acc = perc_model.accuracy(X_test_perc, y_test_perc)
    
    # Time per prediction
    pred_start = time.time()
    _ = perc_model.predict(X_test_perc)
    perc_pred_time = (time.time() - pred_start) / len(X_test_perc)
    
    # ========= APPLES-TO-APPLES COMPARISON ============
    print("\n" + "=" * 60)
    print("APPLES-TO-APPLES COMPARISON (Both as Classifiers)")
    print("=" * 60)
    
    print("\n LINEAR REGRESSION (with Sigmoid):")
    print(f"   Training Accuracy:    {lr_train_acc * 100:.2f}%")
    print(f"   Testing Accuracy:     {lr_test_acc * 100:.2f}%")
    print(f"   Training Time:        {lr_training_time:.4f} seconds")
    print(f"   Time per Prediction:  {lr_pred_time * 1000:.6f} ms")
    print(f"   Optimizer:            Gradient Descent (MSE)")
    
    print("\n PERCEPTRON:")
    print(f"   Training Accuracy:    {perc_train_acc * 100:.2f}%")
    print(f"   Testing Accuracy:     {perc_test_acc * 100:.2f}%")
    print(f"   Training Time:        {perc_training_time:.4f} seconds")
    print(f"   Time per Prediction:  {perc_pred_time * 1000:.6f} ms")
    print(f"   Optimizer:            Perceptron Learning Rule")
    print(f"   Convergence:          {'Early' if len(perc_model.errors) < 1000 else 'Max iterations'}")
    
    print("\n COMPARISON:")
    print(f"   Faster Training:      {'Linear Regression' if lr_training_time < perc_training_time else 'Perceptron'} ({min(lr_training_time, perc_training_time):.4f}s)")
    print(f"   Faster Prediction:    {'Linear Regression' if lr_pred_time < perc_pred_time else 'Perceptron'} ({min(lr_pred_time, perc_pred_time)*1000:.6f}ms)")
    print(f"   Better Accuracy:      {'Linear Regression' if lr_test_acc > perc_test_acc else 'Perceptron'} ({max(lr_test_acc, perc_test_acc)*100:.2f}%)")
    
    accuracy_diff = abs(lr_test_acc - perc_test_acc) * 100
    print(f"   Accuracy Difference:  {accuracy_diff:.2f}%")
    
    print("\n" + "=" * 60)
    
    # Save results
    with open('comparison_results.txt', 'w') as f:
        f.write("APPLES-TO-APPLES MODEL COMPARISON\n")
        f.write("=" * 60 + "\n\n")
        f.write("Both models evaluated as binary classifiers using accuracy metric\n\n")
        f.write(f"Dataset: binary_classification.csv\n")
        f.write(f"Total Samples: {len(X_reg)}\n")
        f.write(f"Training Samples: {len(X_train_reg)}\n")
        f.write(f"Testing Samples: {len(X_test_reg)}\n\n")
        
        f.write("LINEAR REGRESSION (with Sigmoid):\n")
        f.write(f"  Training Accuracy: {lr_train_acc * 100:.2f}%\n")
        f.write(f"  Testing Accuracy: {lr_test_acc * 100:.2f}%\n")
        f.write(f"  Training Time: {lr_training_time:.4f}s\n")
        f.write(f"  Time per Prediction: {lr_pred_time * 1000:.6f}ms\n\n")
        
        f.write("PERCEPTRON:\n")
        f.write(f"  Training Accuracy: {perc_train_acc * 100:.2f}%\n")
        f.write(f"  Testing Accuracy: {perc_test_acc * 100:.2f}%\n")
        f.write(f"  Training Time: {perc_training_time:.4f}s\n")
        f.write(f"  Time per Prediction: {perc_pred_time * 1000:.6f}ms\n")
        f.write(f"  Convergence Iterations: {len(perc_model.errors)}\n\n")
        
        f.write("CONCLUSION:\n")
        f.write(f"Accuracy difference: {accuracy_diff:.2f}%\n")
        f.write(f"Both models perform similarly on this binary classification task.\n")
    
    print(" Results saved to 'comparison_results.txt'")
    
    return {
        'lr_train': lr_train_acc,
        'lr_test': lr_test_acc,
        'lr_time': lr_training_time,
        'perc_train': perc_train_acc,
        'perc_test': perc_test_acc,
        'perc_time': perc_training_time
    }
if __name__ == "__main__":
    results = compare_models()