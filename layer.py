class Layer:
    """
    Abstract class representing layer for neural network
    """
    def __init__(self) -> None:
        self.input = None
        self.output = None

    def forward_propagation(self, input):
        """
        Used to get data from previous layer
        """
        raise NotImplementedError

    def bacwards_propagation(self, output_error, learning_rate):
        """
        Needed to provide derrivative of an error based on input
        """
        raise NotImplementedError