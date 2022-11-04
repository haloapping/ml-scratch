import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from LogisticRegression import LogisticRegression
from classification_metrics import accucary

# load dataset
X, y = load_breast_cancer(return_X_y=True) 

# split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y)

# build and training model
model = LogisticRegression(lr=0.01)
model.fit(X_train, y_train)

# predict
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# evaluate
train_accuracy = accucary(y_train, y_train_pred)
test_accuracy = accucary(y_test, y_test_pred)

print(train_accuracy, test_accuracy)