import numpy as np
import pandas as pd
import json
import time
from linear_regression import LinearRegression
from perceptron import Perceptron


def load_dataset(filepath):
    
    df = pd.read_csv(filepath)
    X = df[['x1', 'x2']].values
    y = df['label'].values
    return X, y


def calculate_accuracy(y_true, y_pred):
    
    return np.mean(y_true == y_pred)


def measure_prediction_time(model, X, n_runs=100):
   
    times = []
    for _ in range(n_runs):
        start = time.time()
        model.predict(X)
        end = time.time()
        times.append(end - start)
    return np.mean(times)


def train_and_evaluate(model, model_name, X, y):
   
    print(f"\nTraining {model_name}...")
    
    start_time = time.time()
    model.fit(X, y)
    convergence_time = time.time() - start_time
    
    y_pred = model.predict(X)
    accuracy = calculate_accuracy(y, y_pred)
    avg_prediction_time = measure_prediction_time(model, X)
    
    metrics = {
        'model': model_name,
        'accuracy': float(accuracy),
        'time_to_convergence': float(convergence_time),
        'iterations_to_converge': int(model.iterations_to_converge),
        'time_per_prediction': float(avg_prediction_time)
    }
    
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Time to convergence: {convergence_time:.6f} seconds")
    print(f"  Iterations to converge: {model.iterations_to_converge}")
    print(f"  Time per prediction: {avg_prediction_time:.6f} seconds")
    
    return metrics


def main():
    
    print("="*70)
    print("Model Comparison: Linear Regression vs Perceptron")
    print("="*70)
    
    dataset_path = '../datasets/binary_classification.csv'
    
    print(f"\nLoading dataset from {dataset_path}...")
    X, y = load_dataset(dataset_path)
    print(f"Loaded {len(X)} samples with {X.shape[1]} features")
    print(f"Label distribution: Class 0: {np.sum(y == 0)}, Class 1: {np.sum(y == 1)}")
   
    all_metrics = []
    
    lr_model = LinearRegression(learning_rate=0.01, max_iterations=1000)
    lr_metrics = train_and_evaluate(lr_model, 'Linear Regression', X, y)
    all_metrics.append(lr_metrics)
    
    perceptron_model = Perceptron(learning_rate=0.01, max_iterations=1000)
    perceptron_metrics = train_and_evaluate(perceptron_model, 'Perceptron', X, y)
    all_metrics.append(perceptron_metrics)
    
    print("\n" + "="*70)
    print("Saving metrics to metrics.json...")
    with open('metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print("✓ Metrics saved successfully!")
    print("="*70)
    
    print("\nSUMMARY COMPARISON:")
    print("-"*70)
    print(f"{'Metric':<25} {'Linear Regression':<20} {'Perceptron':<20}")
    print("-"*70)
    print(f"{'Accuracy':<25} {lr_metrics['accuracy']:<20.4f} {perceptron_metrics['accuracy']:<20.4f}")
    print(f"{'Time to Convergence (s)':<25} {lr_metrics['time_to_convergence']:<20.6f} {perceptron_metrics['time_to_convergence']:<20.6f}")
    print(f"{'Iterations':<25} {lr_metrics['iterations_to_converge']:<20} {perceptron_metrics['iterations_to_converge']:<20}")
    print(f"{'Time per Prediction (s)':<25} {lr_metrics['time_per_prediction']:<20.6f} {perceptron_metrics['time_per_prediction']:<20.6f}")
    print("-"*70)


if __name__ == '__main__':
    main()
