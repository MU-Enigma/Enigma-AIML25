import time
from linear_regression import LinearRegression, load_dataset
from perceptron import Perceptron, load_binary_dataset

def compare_models():
    """
    Compare Linear Regression and Perceptron on the dataset
    Log metrics: Accuracy, Time to convergence, Time per prediction
    """
    print("=" * 60)
    print("MODEL COMPARISON: Linear Regression vs Perceptron")
    print("=" * 60)
    
    # Load datasets
    print("\n[1] Loading datasets...")
    X_reg, y_reg = load_dataset('../../datasets/binary_classification.csv')
    X_perc, y_perc = load_binary_dataset('../../datasets/binary_classification.csv')
    
    # Split data (80/20)
    split_idx_reg = int(0.8 * len(X_reg))
    split_idx_perc = int(0.8 * len(X_perc))
    
    X_train_reg, X_test_reg = X_reg[:split_idx_reg], X_reg[split_idx_reg:]
    y_train_reg, y_test_reg = y_reg[:split_idx_reg], y_reg[split_idx_reg:]
    
    X_train_perc, X_test_perc = X_perc[:split_idx_perc], X_perc[split_idx_perc:]
    y_train_perc, y_test_perc = y_perc[:split_idx_perc], y_perc[split_idx_perc:]
    
    print(f"Dataset size: {len(X_reg)} samples")
    print(f"Training: {len(X_train_reg)} | Testing: {len(X_test_reg)}")
    
    # ========== LINEAR REGRESSION ==========
    print("\n" + "=" * 60)
    print("[2] Training Linear Regression...")
    print("=" * 60)
    
    lr_start = time.time()
    lr_model = LinearRegression(learning_rate=0.0001, iterations=1000)
    lr_model.fit(X_train_reg, y_train_reg)
    lr_training_time = time.time() - lr_start
    
    # Test Linear Regression
    lr_train_score = lr_model.score(X_train_reg, y_train_reg)
    lr_test_score = lr_model.score(X_test_reg, y_test_reg)
    
    # Time per prediction
    pred_start = time.time()
    _ = lr_model.predict(X_test_reg)
    lr_pred_time = (time.time() - pred_start) / len(X_test_reg)
    
    # ========== PERCEPTRON ==========
    print("\n" + "=" * 60)
    print("[3] Training Perceptron...")
    print("=" * 60)
    
    perc_start = time.time()
    perc_model = Perceptron(learning_rate=0.1, iterations=1000)
    perc_model.fit(X_train_perc, y_train_perc)
    perc_training_time = time.time() - perc_start
    
    # Test Perceptron
    perc_train_acc = perc_model.accuracy(X_train_perc, y_train_perc)
    perc_test_acc = perc_model.accuracy(X_test_perc, y_test_perc)
    
    # Time per prediction
    pred_start = time.time()
    _ = perc_model.predict(X_test_perc)
    perc_pred_time = (time.time() - pred_start) / len(X_test_perc)
    
    # ========== RESULTS COMPARISON ==========
    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    
    print("\n LINEAR REGRESSION:")
    print(f"   Training R² Score:    {lr_train_score:.4f}")
    print(f"   Testing R² Score:     {lr_test_score:.4f}")
    print(f"   Training Time:        {lr_training_time:.4f} seconds")
    print(f"   Time per Prediction:  {lr_pred_time * 1000:.6f} ms")
    print(f"   Convergence:          {'Yes' if lr_training_time < 10 else 'Slow'}")
    
    print("\n PERCEPTRON:")
    print(f"   Training Accuracy:    {perc_train_acc * 100:.2f}%")
    print(f"   Testing Accuracy:     {perc_test_acc * 100:.2f}%")
    print(f"   Training Time:        {perc_training_time:.4f} seconds")
    print(f"   Time per Prediction:  {perc_pred_time * 1000:.6f} ms")
    print(f"   Convergence:          {'Yes' if len(perc_model.errors) < 1000 else 'Max iterations'}")
    print(f"   Final Errors:         {perc_model.errors[-1]}")
    
    print("\n WINNER BY CATEGORY:")
    print(f"   Faster Training:      {'Linear Regression' if lr_training_time < perc_training_time else 'Perceptron'}")
    print(f"   Faster Prediction:    {'Linear Regression' if lr_pred_time < perc_pred_time else 'Perceptron'}")
    print(f"   Better Performance:   {'Linear Regression' if lr_test_score > perc_test_acc else 'Perceptron'}")
    
    print("\n" + "=" * 60)
    
    # Save results to file
    with open('comparison_results.txt', 'w') as f:
        f.write("MODEL COMPARISON RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset: binary_classification.csv\n")
        f.write(f"Total Samples: {len(X_reg)}\n")
        f.write(f"Training Samples: {len(X_train_reg)}\n")
        f.write(f"Testing Samples: {len(X_test_reg)}\n\n")
        
        f.write("LINEAR REGRESSION:\n")
        f.write(f"  Training R² Score: {lr_train_score:.4f}\n")
        f.write(f"  Testing R² Score: {lr_test_score:.4f}\n")
        f.write(f"  Training Time: {lr_training_time:.4f}s\n")
        f.write(f"  Time per Prediction: {lr_pred_time * 1000:.6f}ms\n\n")
        
        f.write("PERCEPTRON:\n")
        f.write(f"  Training Accuracy: {perc_train_acc * 100:.2f}%\n")
        f.write(f"  Testing Accuracy: {perc_test_acc * 100:.2f}%\n")
        f.write(f"  Training Time: {perc_training_time:.4f}s\n")
        f.write(f"  Time per Prediction: {perc_pred_time * 1000:.6f}ms\n")
        f.write(f"  Convergence Iterations: {len(perc_model.errors)}\n")
    
    print(" Results saved to 'comparison_results.txt'")
    
    return {
        'lr_train': lr_train_score,
        'lr_test': lr_test_score,
        'lr_time': lr_training_time,
        'perc_train': perc_train_acc,
        'perc_test': perc_test_acc,
        'perc_time': perc_training_time
    }

if __name__ == "__main__":
    results = compare_models()