from typing import TypeVar
from abc import ABC, abstractmethod
T = TypeVar('T')

class Layer(ABC):
    """
    Abstract class representing layer for neural network
    """
    def __init__(self) -> None:
        self.input = None
        self.output = None

    @abstractmethod
    def forward_propagation(self, input: T) -> T:
        """
        Used to get data from previous layer
        """
        pass

    @abstractmethod
    def backward_propagation(self, output_error, learning_rate) -> T:
        """
        Needed to provide derrivative of an error based on input
        """
        pass