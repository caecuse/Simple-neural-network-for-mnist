import numpy as np


# loss function in this case is mean squared error
def mse(y_true: float, y_pred: float) -> float:
    return np.mean(np.power(y_true-y_pred, 2))


def mse_prime(y_true: float, y_pred: float) -> float:
    return 2*(y_pred-y_true)/y_true.size
