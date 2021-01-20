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
try:
    from heuristics import Heuristics
except ModuleNotFoundError:
    from inference.inference_src.heuristics import Heuristics

class MyBattlesnakeHeuristics(Heuristics):
    '''
    The BattlesnakeHeuristics class allows you to define handcrafted rules of the snake.
    '''
    FOOD_INDEX = 0
    def __init__(self):
        pass
       
    @Heuristics.negative_heuristics
    def banned_wall_hits(self, state, snake_id, turn_count, health, json):
        '''
        Heuristics to stop the snakes from hitting a wall.
        '''
        your_snake_body = json["you"]["body"]
        y, x = your_snake_body[0]["y"], your_snake_body[0]["x"]
        height = json["board"]["height"]
        width = json["board"]["width"]
        
        up = y+1 < height
        down = y-1 >= 0
        left = x-1 >= 0
        right = x+1 < width
        
        return [up, down, left, right]
    
    @Heuristics.negative_heuristics
    def banned_forbidden_moves(self, state, snake_id, turn_count, health, json):
        '''
        Heuristics to stop the snakes from forbidden moves.
        '''
        your_snake_body = json["you"]["body"]
        if len(your_snake_body) == 1:
            return [True, True, True, True]
        head_y, head_x = your_snake_body[0]["y"], your_snake_body[0]["x"]
        next_y, next_x = your_snake_body[1]["y"], your_snake_body[1]["x"]
        
        up = not (head_y+1 == next_y and head_x == next_x)
        down = not (head_y-1 == next_y and head_x == next_x)
        left = not (head_y == next_y and head_x-1 == next_x)
        right = not (head_y == next_y and head_x+1 == next_x)

        return [up, down, left, right]
    
    @Heuristics.positive_heuristics
    def go_to_food_if_close(self, state, snake_id, turn_count, health, json):
        '''
        Example heuristic to move towards food if it's close to you.
        '''
        if health[snake_id] > 30:
            return [True, True, True, True]  
        
        # Get the position of the snake head
        your_snake_body = json["you"]["body"]
        y, x = your_snake_body[0]["y"], your_snake_body[0]["x"]
        
        # Get food locations
        food = json["board"]["food"]
        
        for f in food:
            if x==f["x"] and y+1==f["y"]:
                return [True, False, False, False]
            if x==f["x"] and y-1==f["y"]:
                return [False, True, False, False]
            if x-1==f["x"] and y==f["y"]:
                return [False, False, True, False]
            if x+1==f["x"] and y==f["y"]:
                return [False, False, False, True]
        
        return [True, True, True, True]
    
    def run(self, state, snake_id, turn_count, health, json, action):
        '''
        The main function of the heuristics.
        
        Parameters:
        -----------
        `state`: np.array of size (map_size[0]+2, map_size[1]+2, 1+number_of_snakes)
        Provides the current observation of the gym.
        Your target snake is state[:, :, snake_id+1]
        *Note*: This is actually not correct for the simulation. The latter has the 
        form np.array of size [map_size[0], map_size[1], 6] which is build by
        build_state_for_snake() in heuristics_utils.py. To overcome this problem use
        the json format.
    
        `snake_id`: int
        Indicates the id where id \in [0...number_of_snakes]
    
        `turn_count`: int
        Indicates the number of elapsed turns
    
        `health`: dict
        Indicates the health of all snakes in the form of {int: snake_id:int: health}
        
        `json`: dict
        Provides the same information as above, in the same format as the battlesnake engine

        `action`: np.array or list of size 4
        The qvalues of the actions calculated. The 4 values correspond to [up, down, left, right]
        '''
        log_string = ""
        # The default `best_action` to take is the one that provides has the largest Q value.
        # If you think of something else, you can edit how `best_action` is calculated
        action = np.array(action)
        best_action = int(np.argmax(action))
        
        # TODO: Combine heuristics
        # TODO: Be careful with np.argmax on masked action,
        # Q-Values can be negative thus np.argmax prefers
        # illegal moves with 0 over negative valid actions
        wall_masks = self.banned_wall_hits(state, snake_id, turn_count, health, json)
        if best_action not in np.where(wall_masks)[0]:
            log_string += "Hit wall "
            best_action = int(np.argmax(action * np.array(wall_masks)))
        
        forbidden_move_masks = self.banned_forbidden_moves(state, snake_id, turn_count, health, json)
        if best_action not in np.where(forbidden_move_masks)[0]:
            log_string += "Forbidden "
            mask = np.logical_not(forbidden_move_masks) * -1e6
            best_action = int(np.argmax(action * mask))
            
        go_to_food_masks = self.go_to_food_if_close(state, snake_id, turn_count, health, json)
        if best_action not in np.where(go_to_food_masks)[0]:
            log_string += "Food "
            best_action = int(np.argmax(action * np.array(go_to_food_masks)))

        # TODO: add your own heuristics
        if best_action not in [0, 1, 2, 3]:
            best_action = random.choice([0, 1, 2, 3])
        
        return best_action, log_string
