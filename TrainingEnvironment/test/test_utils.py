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

import os
import time

import numpy as np

from battlesnake_gym.snake_gym import BattlesnakeGym
from battlesnake_gym.snake import Snake

def simulate_snake(env, actions, render=False, break_with_done=True):
    '''
    Helper function to run certain actions based in an environment.

    Parameters:
    ----------
    env: BattlesnakeGym
    actions: [np.array(number_of_snakes)]
        A list of np.arrays corresponding to the actions taken at each time step.
        The size of the np.array is equal to the number of snakes in env

    render: Bool, optional, default: False
        Boolean to indicat should the gym be visualised

    break_with_done: Bool, optional, default: True
        By default, BattlesnakeGym will send a done signal if there is only one snake
        remaining (an indication that the game is over). This boolean allows testing to
        occur even if there is only 1 snake

    Returns:
    -------
    observation: np.array
        states of the last timestep taken

    total_reward: int
        the sum of all rewards
    done: bool
    info: {}
        Misc. information
    '''
    total_reward = {}
    for i, action in enumerate(actions):
        if render:
            env.render()
            time.sleep(0.2)
        observation, reward, done, info = env.step(action)
        for k in reward:
            if k not in total_reward: total_reward[k] = 0
            total_reward[k] += reward[k]
        if done:
            if break_with_done:
                break
    return observation, total_reward, done, info

def grow_snake():
    ''''
    Helper function to grow a snake.
    '''
    snake_location = [(0, 0)]
    food_location = [(2, 0), (4, 2), (2, 4), (4, 6), (6, 8),
                     (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
                     (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
                     (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]
    env = BattlesnakeGym(map_size=(9, 9), number_of_snakes=1,
                         snake_spawn_locations=snake_location,
                         food_spawn_locations=food_location,
                         verbose=True)

    env.food.max_turns_to_next_food_spawn = 2  # Hack to make sure that food is spawned every turn

    actions = [[Snake.DOWN], [Snake.DOWN],
               [Snake.RIGHT], [Snake.RIGHT], [Snake.DOWN], [Snake.DOWN],
               [Snake.RIGHT], [Snake.RIGHT], [Snake.UP], [Snake.UP],
               [Snake.RIGHT], [Snake.RIGHT], [Snake.DOWN], [Snake.DOWN],
               [Snake.RIGHT], [Snake.RIGHT], [Snake.DOWN], [Snake.DOWN], [Snake.DOWN]]

    simulate_snake(env, actions, render=should_render(), break_with_done=False)
    return env

def grow_two_snakes(snake_starting_positions):
    '''
    Helper function to grow two snakes based on the snake_starting_position.
    '''
    snake_location = snake_starting_positions
    food_location = [(2, 0), (2, 2), (4, 2), (2, 4), (4, 6),
                     (7, 5), (7, 4), (7, 3), (7, 2)] + [(0, 0)] * 100
    env = BattlesnakeGym(map_size=(9, 9), number_of_snakes=2,
                         snake_spawn_locations=snake_location,
                         food_spawn_locations=food_location,
                         verbose=True)
    env.food.max_turns_to_next_food_spawn = 2  # Hack to make sure that food is spawned every turn

    actions_snake1 = [[Snake.DOWN], [Snake.DOWN],
                      [Snake.RIGHT], [Snake.RIGHT], [Snake.DOWN], [Snake.DOWN],
                      [Snake.RIGHT], [Snake.RIGHT], [Snake.UP], [Snake.UP],
                      [Snake.RIGHT], [Snake.RIGHT], [Snake.DOWN], [Snake.DOWN],
                      [Snake.RIGHT], [Snake.RIGHT], [Snake.DOWN], [Snake.DOWN], [Snake.DOWN]]

    actions_snake2 = [[Snake.RIGHT] ] *7 + [[Snake.DOWN]] + [[Snake.LEFT] ] *7 + [[Snake.DOWN]] + [[Snake.RIGHT] ] *3
    tmp_actions = list(zip(actions_snake1, actions_snake2))
    actions = []
    for action in tmp_actions:
        actions.append(np.array([action[0], action[1]]))

    simulate_snake(env, actions, render=should_render(), break_with_done=False)
    return env

def should_render():
    '''
    Helper function to know whether to render the game based on env var
    BATTLESNAKE_RENDER=1 or BATTLESNAKE_RENDER=0. Default 0
    :return:
    '''
    if 'BATTLESNAKE_RENDER' in os.environ:
        return int(os.environ.get('BATTLESNAKE_RENDER')) == 1
    else:
        return False