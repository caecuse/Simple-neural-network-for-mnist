from layer import Layer
import numpy as np


class CLayer(Layer):
    """
    Represents "static" layer
    """
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.random.randn(1, output_size)

    def forward_propagation(self, input_data):
        """
        returns output for a given input
        """
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward_propagation(self, output_error, learning_rate: np.ndarray) -> np.ndarray:
        """
        Computes dE/dW, dE/dB for a given output_error = dE/dY. Returns input_error = dE/dX
        """
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        # update parameters using gradient descent with learning rate
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error
