import json

# Load metrics from JSON file
with open('metrics.json') as f:
    metrics = json.load(f)

# Prepare the comparison content
comparison_content = """
Model Performance Comparison
============================

Accuracy:
- Linear Regression: {:.2f}
- Perceptron: {:.2f}

Time to Convergence:
- Linear Regression: {:.4f} seconds
- Perceptron: {:.4f} seconds

Average Time per Prediction:
- Linear Regression: {:.6f} seconds
- Perceptron: {:.6f} seconds

Observations:
- Both models achieved high accuracy (~{:.2f}), indicating the dataset might be linearly separable.
- The time to convergence varies based on the dataset and the model's complexity.
- Typically, the perceptron converges faster on linearly separable data, whereas linear regression might take longer depending on learning rate.

Analysis:
<Add your detailed interpretation here based on the above metrics, explaining why models performed as they did, and noting any anomalies or insights.>
""".format(
    metrics['LinearRegression']['accuracy'],
    metrics['Perceptron']['accuracy'],
    metrics['LinearRegression']['time_to_convergence'],
    metrics['Perceptron']['time_to_convergence'],
    metrics['LinearRegression']['time_per_prediction'],
    metrics['Perceptron']['time_per_prediction'],
    metrics['LinearRegression']['accuracy'] * 100,
    # Add any other insights or comparisons you'd like
)

# Write the comparison and analysis to analysis.txt
with open('analysis.txt', 'w') as f:
    f.write(comparison_content)
    f.write("\nDetailed Analysis:\n")
    f.write("<Your detailed explanation about why one performed better or worse, etc.>\n")
