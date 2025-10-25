# Analysis: Linear Regression vs Perceptron

## Dataset 1: Linearly Separable
- Both models performed well.
- Perceptron achieved slightly higher accuracy (0.88) but took longer to train (0.52s vs 0.007s).
- Linear Regression converged much faster but outputs continuous predictions, hence slightly lower accuracy.

## Dataset 2: Non-Linearly Separable
- Surprisingly, Linear Regression slightly outperformed the Perceptron.
- This happens because the Perceptron can’t adapt to non-linear boundaries.
- Linear Regression’s continuous nature helps it approximate curved boundaries somewhat better.

## Summary
- **Speed:** Linear Regression was much faster.
- **Accuracy:** Comparable results, Perceptron wins on linear data.
- **Generalization:** Linear Regression performed more consistently across datasets.
