import numpy as np

class MyBattlesnakeHeuristics:
    '''
    The BattlesnakeHeuristics class allows you to define handcrafted rules of the snake.
    '''
    def __init__(self):
        pass
    
    def run(self, state, snake_id, turn_count, health, action):
        '''
        '''
        # TO DO, add your own heuristics
        
        action = np.argmax(action)
        assert action in [0, 1, 2, 3], "{} is not a valid action.".format(action)
        return action