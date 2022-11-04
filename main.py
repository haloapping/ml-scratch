import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from LinearRegression import LinearRegression
from regression_metrics import mse, r2_score

# load dataset
X, y = make_regression(n_samples=100, n_features=1, noise=10)

# split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

# build and training model
model = LinearRegression(lr=0.1)
model.fit(X_train, y_train)

# predict
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# evaluate
train_mse = mse(y_train, y_train_pred)
train_r2_score = r2_score(y_train, y_train_pred)
test_mse = mse(y_test, y_test_pred)
test_r2_score = r2_score(y_test, y_test_pred)

# visualize
plt.title(f"Train MSE: {train_mse:.3f} | Train $R^2$: {train_r2_score:.3f} | Test MSE : {test_mse:.3f} | Test $R^2$: {test_r2_score:.3f} ")
plt.scatter(X_train, y_train, label="train")
plt.plot(X_train, y_train_pred, label="train_pred")
plt.scatter(X_test, y_test, label="test")
plt.plot(X_test, y_test_pred, label="test_pred")
plt.legend()
plt.grid()
plt.show()