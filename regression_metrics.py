import numpy as np

def r2_score(y_true, y_pred):
    mean_y_true = np.mean(y_true)
    ss_res = np.sum(np.square(y_true - y_pred))
    ss_tot = np.sum(np.square(y_true - mean_y_true))

    return 1 - (ss_res / ss_tot)

def mse(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))