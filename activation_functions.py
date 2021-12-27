import numpy as np


# activation function and its derivative
def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x: float) -> float:
    return sigmoid(x) * (1 - sigmoid(x))
