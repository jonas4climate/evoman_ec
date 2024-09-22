from evoman.controller import Controller
import numpy as np


class controller_pablo(Controller):
	"""
	Fully connected network with custom input features:
	Uses 2 custom input features to reduce the network input space from 20 to 18:
	Substitutes inputs 16 and 17 with the distance between player and enemy
	Substitutes inputs 18 and 19 with a flag for whether the player is facing the enemy (+-1)
	"""

	def __init__(self):
		self.weights = None
	
	def set_weights(self, weights, n_hidden):
		self.weights = weights
		self.n_hidden = n_hidden
		self.n_inputs = 20
		self.n_outputs = 5
	
    # only important method, add more for utility but this is the one called by the game
	def control(self, inputs, controller):
		assert self.weights is not None, "Weights have not been set."

		# Compute the 2 custom features
		distance_player_enemy = np.sqrt(inputs[16]**2 + inputs[17]**2)
		looking_at_enemy = 1 if ((inputs[18] and inputs[19]) or (inputs[18] and not inputs[19])) else -1
		
		inputs_2 = np.zeros(self.n_inputs)
		inputs_2[0:16] = inputs[0:16]
		inputs_2[16] = distance_player_enemy
		inputs_2[17] = looking_at_enemy
		inputs = inputs_2
		
		
		# Create custom input matrix with the first 16 inputs and the 2 custom features
		#inputs = np.array(inputs[0:16].append(distance_player_enemy).append(looking_at_enemy))
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