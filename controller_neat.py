from evoman.environment import Environment
from evoman.controller import Controller

class controller_neat(Controller):
	def __init__(self, log_history=False):
		self.decision_history = []
		self.log_history = log_history

	def reset_history(self):
		self.decision_history = []
	
    # only important method, add more for utility but this is the one called by the game
	def control(self, inputs, controller):
		inputs = (inputs-min(inputs))/float((max(inputs)-min(inputs))) # normalize inputs

		action = controller.activate(inputs) # Get action (or prediction) from the network

		left, right, jump, shoot, release = 0, 0, 0, 0, 0

		# Process inputs and make decisions by adjusting values below before returning
		if action[0] > 0.5:
			left = 1
		
		if action[1] > 0.5:
			right = 1
		
		if action[2] > 0.5:
			jump = 1

		if action[3] > 0.5:
			shoot = 1
		
		if action[4] > 0.5:
			release = 1

		decision = [left, right, jump, shoot, release]
		
		if self.log_history:
			self.decision_history.append(decision + [left or right] + [jump or release])
		
		return decision