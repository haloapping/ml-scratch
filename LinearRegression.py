import numpy as np
from tqdm import tqdm

class LinearRegression:
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, x, y):
        n_samples, n_features = x.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in tqdm(range(self.n_iters)):
            # Predict
            y_pred = (x @ self.weights) + self.bias
            
            # Calculate gradient
            dw = (1 / n_samples) * (x.T @ (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Update weights and bias
            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    def predict(self, x):
        return x @ self.weights + self.bias