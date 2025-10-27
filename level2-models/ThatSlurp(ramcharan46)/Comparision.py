from LinReg_GD import *
from Perceptron import perceptron
import numpy as np
from typing import Tuple

def create_synthetic_data(num_samples: int = 100,true_m: int = 2,true_c: int = 1,noise_level: int = 5,) -> Tuple[np.array, np.array]:
    X = np.linspace(0, 10, num_samples)
    Y_true = true_m * X + true_c
    noise = np.random.normal(0, noise_level, num_samples)
    Y = Y_true + noise
    return X, Y

def train_model_LRGD(X, Y, epochs=1000, learning_rate=0.01):
    print("\nStarting Training Loop (Gradient Descent)")
    m = np.random.randn()
    c = np.random.randn()
    print(f"Initial parameters: m={m:.4f}, c={c:.4f}")
    loss_history = []
    for epoch in range(1, epochs + 1):
        m, c, dm, dc = gradient_descent(X, Y, m, c, learning_rate)
        Y_pred = predict(X, m, c)
        loss = mean_squared_error(Y, Y_pred)
        loss_history.append(loss)
        if epoch % 1 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{epochs} | Loss: {loss:.4f} | New m: {m:.4f}, New c: {c:.4f}")
    return m, c, loss_history

def train_model_P(X, Y, epochs=1000, learning_rate=0.01):
    print("\nStarting Training Loop (Perceptron)")
    weights = np.random.randn(X.shape[1])
    bias = np.random.randn()
    print(f"Initial parameters: weights={weights}, bias={bias:.4f}")
    for epoch in range(1, epochs + 1):
        Y_pred = perceptron(X, weights, bias)
        errors = Y - Y_pred
        weights += learning_rate * np.dot(X.T, errors)
        bias += learning_rate * np.sum(errors)
        if epoch % 100 == 0 or epoch == 1:
            loss = np.mean(errors ** 2)
            print(f"Epoch {epoch:4d}/{epochs} | Loss: {loss:.4f} | Weights: {weights}, Bias: {bias:.4f}")
    return weights, bias

X_data, Y_data = create_synthetic_data()

final_m, final_c, loss_history = train_model_LRGD(X_data, Y_data, epochs=500, learning_rate=0.01)
X_data_perceptron = X_data.reshape(-1, 1)
final_weights, final_bias = train_model_P(X_data_perceptron, (Y_data > np.mean(Y_data)).astype(int), epochs=500, learning_rate=0.01)

# Plotting the results for both linear regression and perceptron :)
import matplotlib.pyplot as plt
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.scatter(X_data, Y_data, color='blue', label='Data Points')
plt.plot(X_data, predict(X_data, final_m, final_c), color='red', label='Fitted Line')
plt.title('Linear Regression with Gradient Descent')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.subplot(1, 2, 2)
plt.scatter(X_data, (Y_data > np.mean(Y_data)).astype(int), color='blue', label='Data Points')
plt.scatter(X_data, perceptron(X_data_perceptron, final_weights, final_bias), color='red', label='Perceptron Output')
plt.title('Perceptron Classification')
plt.xlabel('X')
plt.ylabel('Class')
plt.legend()
plt.show()





