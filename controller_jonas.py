from evoman.environment import Environment
from evoman.controller import Controller

class controller_jonas(Controller):
	
    # only important method, add more for utility but this is the one called by the game
	def control(self, inputs, controller):
		# Process inputs and make decisions by adjusting values below before returning
		
		left = 0
		right = 0
		jump = 1
		shoot = 1
		release = 0
		
		return [left, right, jump, shoot, release]