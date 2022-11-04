import numpy as np

def accucary(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)