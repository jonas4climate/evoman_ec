from evoman.environment import Environment
from evoman.controller import Controller

class controller_pablo_francijn(Controller):
	
    # only important method, add more for utility but this is the one called by the game
	def control(self, inputs, controller):
		inputs = (inputs-min(inputs))/float((max(inputs)-min(inputs))) # normalize inputs

		action = controller.activate(inputs)      # Get action (or prediction) from the network
		# Process inputs and make decisions by adjusting values below before returning
		if action[0] > 0.5:
			left = 1
		else:
			left = 0
		
		if action[1] > 0.5:
			right = 1
		else:
			right = 0
		
		if action[2] > 0.5:
			jump = 1
		else:
			jump = 0

		if action[3] > 0.5:
			shoot = 1
		else:
			shoot = 0
		
		if action[4] > 0.5:
			release = 1
		else:
			release = 0
		
		return [left, right, jump, shoot, release]