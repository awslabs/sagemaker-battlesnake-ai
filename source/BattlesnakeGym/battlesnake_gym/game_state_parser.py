# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# or in the "license" file accompanying this file. This file is distributed 
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either 
# express or implied. See the License for the specific language governing 
# permissions and limitations under the License.

from .snake import Snakes
from .food import Food
import pandas as pd
import string
import numpy as np

class Game_state_parser:
    '''
    Class to initialise the gym from a dictionary.
    
    Parameters:
    ----------
    board_dict: dict
        Dictionary to indicate the initial game state
        Dict is in the same form as in the battlesnake engine
        https://docs.battlesnake.com/references/api
    '''
    def __init__(self, game_dict):
        self.game_dict = game_dict
        self.board_dict = self.game_dict["board"]
        self.map_size = (self.board_dict["height"], self.board_dict["width"]) 
        self.number_of_snakes = len(self.board_dict["snakes"])
                
    def parse(self):     
        # Get food locations
        food_locations = []
        for food_location in self.board_dict["food"]:
            x, y = food_location["x"], food_location["y"]
            food_locations.append((y, x))
                    
        food = Food.make_from_list(self.map_size, food_locations)
        snakes = Snakes.make_from_dict(self.map_size, self.board_dict["snakes"])

        turn_count = self.game_dict["turn"]

        return snakes, food, turn_count
