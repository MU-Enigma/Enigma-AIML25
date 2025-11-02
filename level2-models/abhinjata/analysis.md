# Level 2 Model Analysis: Linear Regression vs. Perceptron

This report analyzes the performance of a from-scratch **Linear Regression** model and a **Perceptron** model on two datasets: one linearly separable (`binary_classification.csv`) and one non-linearly separable (`binary_classification_non_lin.csv`).

## 1. Final Metrics

The following metrics were logged from `compareOneLayer.py` after running 1000 epochs with a learning rate of 0.01.

| **Dataset** | **Model** | **Accuracy** | **Convergence Time (sec)** | **Time per Prediction (sec)** |
|:---|:---|:---:|:---:|:---:|
| `binary_classification.csv` (Linear) | **Linear Regression** | **87.17%** | **0.0256** | ~0.0 |
| `binary_classification.csv` (Linear) | Perceptron | 84.33% | 2.6143 | ~0.0 |
| `binary_classification_non_lin.csv` (Non-Linear) | Linear Regression | 79.13% | 0.0250 | ~0.0 |
| `binary_classification_non_lin.csv` (Non-Linear) | **Perceptron** | **79.63%** | **3.6967** | ~1.25e-06 |

---

## 2. Analysis of Results

There are two major stories in these results: the **model's effectiveness (Accuracy)** and the **algorithm's speed (Convergence Time)**.

### Insight 1: Accuracy Analysis

This is the most important part of the analysis. Both models are **linear classifiers**. This means they can *only* learn to draw a single straight line (a "linear boundary") to separate the data.

* **On the `binary_classification.csv` (Linear Data):**
  * Both models performed reasonably well (87% and 84%). This dataset is *mostly* linearly separable.
  * **Surprising Result:** Linear Regression (a regression model!) slightly *outperformed* the Perceptron (a classification model!). This is because Linear Regression's goal is to minimize **Mean Squared Error (MSE)**, while the Perceptron's goal is to minimize **misclassifications**. For this specific dataset, the line that minimized MSE *also* happened to be slightly better at classifying points (after we applied a 0.5 threshold) than the line the Perceptron found.
  * The Perceptron's 84% (and not 100%) suggests the data isn't *perfectly* separable, or that the learning rate/epochs weren't perfectly tuned to find the optimal line.

* **On the `binary_classification_non_lin.csv` (Non-Linear Data):**
  * **Both models failed equally.** Their accuracy dropped to ~79% for both.
  * This is the expected result. This dataset is likely arranged in a curve (like a circle or a "moons" shape). A linear model is *fundamentally incapable* of drawing a curve.
  * The ~79% accuracy they found is the "best possible" straight line they could draw, which still results in misclassifying over 20% of the data. This perfectly demonstrates the limitations of a linear model.

### Insight 2: Speed Analysis

* **Linear Regression Training:** Incredibly fast, at **~0.025 seconds**.
* **Perceptron Training:** Incredibly slow, at **~2.6 to 3.7 seconds**. (Over 100x slower)

This has everything to do with their implementation and architecture.

* **Linear Regression** (in `OneLayerLinearRegressor.py`) uses Gradient Descent with `numpy`. Its calculations (`np.dot(X.T, dy_pred)`) are **vectorized**. This means it performs one single, massive matrix operation to update the weights for all samples at once. This is extremely fast and efficient. [cite: 1LayerLinearRegressor.py]
* **Perceptron** (in `OneLayerClassifierPerceptron.py`) uses the traditional Perceptron learning rule, which **loops over every single sample** (`for idx in range(n_samples):`) inside the `epochs` loop. Python `for` loops are notoriously slow. This iterative, one-by-one update is the source of the 3-second training time. [cite: 1LayerClassifierPerceptron.py]

### Insight 3: Prediction Speed

* Both models are **practically instantaneous** at making predictions.
* Once trained, both models just need to perform one single dot product (`np.dot(X, self.w) + self.b`) to make predictions for the entire dataset. This is a highly optimized vectorized operation, which is why the time is near-zero.

---

## 3. Conclusion (Theory vs. Practice)

This experiment clearly shows the connection between a model's theory and its practical performance:

1. **Theory:** A Perceptron is designed for binary classification. Linear Regression is designed for regression.
   **Practice:** The Perceptron's learning rule (sample-by-sample) is very slow. Linear Regression (using vectorized gradient descent) is much faster to train. Ironically, the regression model (with a threshold) can sometimes be a better classifier than the "correct" classification model, depending on the data.

2. **Theory:** Both models are linear. They work by finding an optimal straight line.
   **Practice:** As expected, both models performed reasonably on linear data but failed on non-linear data. They are the wrong tool for that job. This proves that to solve a non-linear problem, we would need a non-linear model (like a 2-layer Neural Net, SVM with a kernel, or a Decision Tree).

Credit: Gemini 2.5 Pro for documenting based on the core analysis 
