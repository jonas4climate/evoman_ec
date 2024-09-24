import numpy as np
from evoman.controller import Controller

class controller_jonas(Controller):
    def __init__(self):
        self.weights = None

    def set_weights(self, weights, n_hidden):
        self.weights = weights
        self.n_hidden = n_hidden
        self.n_inputs = 20
        self.n_outputs = 5

        hidden_weights_size = self.n_inputs * self.n_hidden
        hidden_bias_size = self.n_hidden
        output_weights_size = self.n_hidden * self.n_outputs
        output_bias_size = self.n_outputs

        expected_size = (hidden_weights_size + hidden_bias_size +
                         output_weights_size + output_bias_size)
        assert len(weights) == expected_size, f"Expected weights array of size {expected_size}, but got {len(weights)}"

        self.hidden_layer_weights = np.array(weights[:hidden_weights_size]).reshape((self.n_inputs, self.n_hidden))
        self.hidden_layer_bias = np.array(weights[hidden_weights_size:hidden_weights_size + hidden_bias_size])
        self.output_layer_weights = np.array(weights[hidden_weights_size + hidden_bias_size:
                                                     hidden_weights_size + hidden_bias_size + output_weights_size]).reshape((self.n_hidden, self.n_outputs))
        self.output_layer_bias = np.array(weights[-output_bias_size:])

    def control(self, inputs, controller):
        assert self.weights is not None, "Weights have not been set."

        inputs = np.array(inputs)
        assert len(inputs) == self.n_inputs, f"Expected {self.n_inputs} inputs, but got {len(inputs)}. This shouldn't happen."

        # Forward pass through the hidden layer
        hidden_layer_output = np.tanh(np.dot(inputs, self.hidden_layer_weights) + self.hidden_layer_bias)

        # Forward pass through the output layer
        output = np.tanh(np.dot(hidden_layer_output, self.output_layer_weights) + self.output_layer_bias)

        # Convert the output to control signals
        left = 1 if output[0] > 0 else 0
        right = 1 if output[1] > 0 else 0
        jump = 1 if output[2] > 0 else 0
        shoot = 1 if output[3] > 0 else 0
        release = 1 if output[4] > 0 else 0

        return [left, right, jump, shoot, release]