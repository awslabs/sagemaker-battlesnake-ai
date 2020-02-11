import numpy as np

class Rewards:
    '''
    Base class to set up rewards for the battlesnake gym
    '''
    def get_reward(self, name, snake_id, episode):
        raise NotImplemented()

class SimpleRewards(Rewards):
    '''
    Simple class to handle a fixed reward scheme
    '''
    def __init__(self):
        self.reward_dict = {"another_turn": 1,
                            "ate_food": 0,
                            "won": 2,
                            "died": -3,
                            "ate_another_snake": 0,
                            "hit_wall": 0,
                            "hit_other_snake": 0,
                            "hit_self": 0,
                            "was_eaten": 0,
                            "other_snake_hit_body": 0,
                            "forbidden_move": -1,
                            "starved": 0}

    def get_reward(self, name, snake_id, episode):
        return self.reward_dict[name]
