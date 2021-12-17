from layer import Layer


# inherit from base class Layer
class ActiveLayer(Layer):
    """
    Represents "thinking" layer
    """
    def __init__(self, activation, activation_prime):
        """
        Sets function used to activate neurons for this layer
        """
        self.activation = activation
        self.activation_prime = activation_prime

    def forward_propagation(self, input_data):
        """
        returns the activated input
        """
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        """
        Returns input_error dE/dX for a given output_error = dE/dY.
        """
        return self.activation_prime(self.input) * output_error
