from layer import Layer
from typing import Callable, List
import numpy as np

class Network:
    def __init__(self) -> None:
        self.layers = []
        self.loss = None
        self.loss_prime = None

    # add layer to network
    def add(self, layer: Layer) -> None:
        self.layers.append(layer)

    # set loss to use
    def use(self, loss: Callable[[float], float], loss_prime: Callable[[float], float]) -> None:
        self.loss = loss
        self.loss_prime = loss_prime

    # predict output for given input
    def predict(self, input_data: List[np.ndarray]) -> List[np.ndarray]:
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    # train the network
    def fit(self, x_train: np.ndarray, y_train: np.ndarray, epochs: int,
            learning_rate: float) -> None:
        # sample dimension first
        samples = len(x_train)

        # training loop
        for i in range(epochs):
            err = 0
            for j in range(samples):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss (for display purpose only)
                err += self.loss(y_train[j], output)

                # backward propagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            # calculate average error on all samples
            err /= samples
            print(f'epoch {i+1}/{epochs}   error={err}')
