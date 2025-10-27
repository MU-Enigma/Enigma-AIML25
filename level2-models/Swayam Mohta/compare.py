import json
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from linear_regression import LinearRegressionGD
from perceptron import Perceptron

def evaluate_model(model, X_train, y_train, X_test, y_test):
    start_fit = time.time()
    model.fit(X_train, y_train)
    fit_time = model.training_time or (time.time() - start_fit)

    start_pred = time.time()
    y_pred = model.predict(X_test)
    pred_time = (time.time() - start_pred) / len(X_test)

    # Round continuous predictions for accuracy
    if y_pred.ndim > 1:
        y_pred = y_pred.flatten()
    y_pred_binary = np.where(y_pred >= 0.5, 1, 0)

    accuracy = accuracy_score(y_test, y_pred_binary)

    return accuracy, fit_time, pred_time


def generate_synthetic_datasets():
    """Creates two simple datasets for demonstration"""
    from sklearn.datasets import make_classification, make_moons

    # Linearly separable
    X1, y1 = make_classification(
        n_samples=500, n_features=2, n_classes=2, n_redundant=0, n_clusters_per_class=1, random_state=42
    )

    # Non-linearly separable
    X2, y2 = make_moons(n_samples=500, noise=0.2, random_state=42)

    return {"linear": (X1, y1), "nonlinear": (X2, y2)}


def main():
    datasets = generate_synthetic_datasets()
    results = {}

    for name, (X, y) in datasets.items():
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        results[name] = {}

        for model_class in [LinearRegressionGD, Perceptron]:
            model = model_class(lr=0.01, epochs=1000)
            acc, train_time, time_per_pred = evaluate_model(model, X_train, y_train, X_test, y_test)

            results[name][model_class.__name__] = {
                "accuracy": round(acc, 4),
                "train_time_sec": round(train_time, 4),
                "time_per_prediction_sec": round(time_per_pred, 6),
            }

    # Save metrics
    with open("metrics.json", "w") as f:
        json.dump(results, f, indent=4)

    print("Results saved to metrics.json")
    print(json.dumps(results, indent=4))


if __name__ == "__main__":
    main()
