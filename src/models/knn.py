# K-Nearest Neighbors from scratch

import numpy as np
from collections import Counter
from .base_model import BaseModel

class KNN(BaseModel):
    """
    K-Nearest Neighbors classifier.
    """
    def __init__(self, k=3):
        """
        Initializes the K-Nearest Neighbors classifier.

        Args:
            k (int): The number of neighbors to use for classification.
        """
        super().__init__(_forward_has_training_logic=False)
        self.k = k

    def fit(self, X, y):
        """
        "Trains" the KNN model by storing the training data.

        Args:
            X (np.ndarray): Training vectors, where n_samples is the number of samples and
                            n_features is the number of features.
            y (np.ndarray): Target values.
        """
        self.X_train = X
        self.y_train = y
        self.hard_set_trained(True)

    def forward(self, X):
        """
        Predicts the class labels for the given data. This is a wrapper around the predict method.

        Args:
            X (np.ndarray): The input samples.

        Returns:
            np.ndarray: The predicted class labels.
        """
        return self.predict(X)

    def predict(self, X):
        """
        Predicts the class labels for the given data.

        Args:
            X (np.ndarray): The input samples.

        Returns:
            np.ndarray: The predicted class labels.
        """
        assert self.is_trained, "Call .fit() before .predict()"
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        # Compute Euclidean distances
        distances = [np.sqrt(np.sum((x_train - x)**2)) for x_train in self.X_train]
        
        # Get k nearest samples, labels
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # Majority vote for classification
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
