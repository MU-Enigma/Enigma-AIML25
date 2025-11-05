import pandas as pd
import numpy as np
import json
import time
import os 

from OneLayerLinearRegressor import LinearRegression
from OneLayerClassifierPerceptron import Perceptron

def calculate_accuracy(y_true, y_pred):
    """Calculates the accuracy of the predictions."""
    return np.mean(y_true == y_pred)

def main():

    # --- Get the path to the directory this script is in ---
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # --- 1. Define Datasets and Models ---
    dataset_files = [
        'binary_classification.csv', 
        'binary_classification_non_lin.csv'
    ]
    
    LR = 0.01
    EPOCHS = 1000
    
    results = {}

    # --- 2. Run All Experiments ---
    for dataset_name in dataset_files:
        print(f"\n--- Testing on Dataset: {dataset_name} ---")
        results[dataset_name] = {}
        
        # --- FIX 3: Create the full, absolute path to the CSV file ---
        file_path = os.path.join(script_dir, dataset_name)
        
        # Load the dataset using the full path
        data = pd.read_csv(file_path)
        X = data[['x1', 'x2']].values
        y = data['label'].values
        
        # --- Run Linear Regression ---
        print("  Training LinearRegression...")
        model_lr = LinearRegression(lr=LR, epochs=EPOCHS)
        
        model_lr.fit(X, y)
        conv_time_lr = model_lr.convergence_time
        
        y_pred_lr, total_pred_time_lr = model_lr.predict(X)
        
        y_pred_class_lr = np.where(y_pred_lr >= 0.5, 1, 0)
        acc_lr = calculate_accuracy(y, y_pred_class_lr)
        time_per_pred_lr = total_pred_time_lr / len(y)
        
        results[dataset_name]["linear_regression"] = {
            "accuracy": acc_lr,
            "convergence_time_sec": conv_time_lr,
            "time_per_prediction_sec": time_per_pred_lr
        }

        # --- Run Perceptron ---
        print("  Training Perceptron...")
        model_p = Perceptron() 
        
        model_p.fit(X, y, lr=LR, epochs=EPOCHS)
        conv_time_p = model_p.convergence_time
        
        y_pred_p, total_pred_time_p = model_p.predict(X)
        
        acc_p = calculate_accuracy(y, y_pred_p)
        time_per_pred_p = total_pred_time_p / len(y)

        results[dataset_name]["perceptron"] = {
            "accuracy": acc_p,
            "convergence_time_sec": conv_time_p,
            "time_per_prediction_sec": time_per_pred_p
        }

    # --- 7. Save Results to JSON ---
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\n--- All experiments complete. Results saved to results.json ---")

if __name__ == "__main__":
    main()