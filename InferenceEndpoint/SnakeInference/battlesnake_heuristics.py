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

import numpy as np
import random

class MyBattlesnakeHeuristics:
    '''
    The BattlesnakeHeuristics class allows you to define handcrafted rules of the snake.
    '''
    def __init__(self):
        pass
    
    def run(self, state, snake_id, turn_count, health, action):
        '''
        The main function of the heuristics.
        
        Parameters:
        -----------
        `state`: np.array of size (3, map_size[0]+2, map_size[1]+2)
        Provides the current observation of the gym
    
        `snake_id`: int
        Indicates the id where id \in [0...number_of_snakes]
    
        `turn_count`: int
        Indicates the number of elapsed turns
    
        `health`: dict
        Indicates the health of all snakes in the form of {snake_id: health}
        TODO: This is not implemented like that for now !!!

        `action`: np.array of size 4
        The qvalues of the actions calculated. The 4 values correspond to [up, down, left, right]
        '''
        # The default `best_action` to take is the one that provides has the largest Q value.
        # If you think of something else, you can edit how `best_action` is calculated
        best_action = np.argmax(action)

        # TO DO, add your own heuristics
        i,j = np.unravel_index(np.argmax(state[:,:,1], axis=None), state[:,:,1].shape)
        snakes = state[:,:,1:].sum(axis=2)
        food = state[:,:,0]
        possible = []
        food_locations = []
        if snakes[i-1,j] == 0:
            possible.append(0) # up
        if snakes[i+1,j] == 0:
            possible.append(1) # down
        if snakes[i,j-1] == 0:
            possible.append(2) # left
        if snakes[i,j+1] == 0:
            possible.append(3) # right

        # Food locations
        if food[i-1,j] == 1:
            food_locations.append(0) # up
        if food[i+1, j] == 1:
            food_locations.append(1) # down
        if food[i,j-1] == 1:
            food_locations.append(2) # left
        if food[i,j+1] == 1:
            food_locations.append(3) # right

        choice = best_action

        if best_action in possible:
            # Don't starve if possible
            if health < 30 and len(food_locations) > 0 and best_action not in food_locations:
                print("eating food instead of move")
                choice = random.choice(food_locations)
        elif len(possible) > 0:
            # Don't starve if possible        
            if health < 30 and len(food_locations) > 0 and best_action not in food_locations:
                print("eating food instead of dying")
                choice = random.choice(food_locations)
            # Don't kill yourself
            else:
                print("Move "+best_action+" is not possible")
                choice = random.choice(possible)

        assert choice in [0, 1, 2, 3], "{} is not a valid action.".format(choice)
        return choice