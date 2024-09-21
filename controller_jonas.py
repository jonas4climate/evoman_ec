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

    def control(self, inputs, controller):
        assert self.weights is not None, "Weights have not been set."

        inputs = np.array(inputs)
        assert len(inputs) == self.n_inputs, f"Expected {self.n_inputs} inputs, but got {len(inputs)}. This shouldn't happen."

        expected_size = self.n_inputs * self.n_inputs + self.n_inputs * self.n_outputs
        assert len(self.weights) == expected_size, f"Expected weights array of size {expected_size}, but got {len(self.weights)}"

        hidden_layer_weights = np.array(self.weights[:self.n_inputs * self.n_inputs]).reshape((self.n_inputs, self.n_inputs))
        output_layer_weights = np.array(self.weights[self.n_inputs * self.n_inputs:]).reshape((self.n_inputs, self.n_outputs))

        hidden_layer_output = np.tanh(np.dot(inputs, hidden_layer_weights))
        output = np.tanh(np.dot(hidden_layer_output, output_layer_weights))

        # Convert the output to control signals
        left = 1 if output[0] > 0 else 0
        right = 1 if output[1] > 0 else 0
        jump = 1 if output[2] > 0 else 0
        shoot = 1 if output[3] > 0 else 0
        release = 1 if output[4] > 0 else 0

        return [left, right, jump, shoot, release]