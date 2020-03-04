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


from battlesnake_gym.snake_gym import BattlesnakeGym
from battlesnake_gym.snake import Snake

from .test_utils import simulate_snake


def test_gym_performance(map_sizes, number_of_snakes):
    """
    Measure the performance of the gym in steps per seconds

    :param map_sizes: [()]
    :param number_of_snakes: []
    :return:
    """

    max_turns_num = 100
    for map_size in map_sizes:
        for num_snakes in number_of_snakes:
            # Create the snakes one after each others
            snake_locations = []
            i = 0
            j = 0
            for _ in range(num_snakes):
                snake_locations.append([i, j])
                if j >= map_size[1] - 2:
                    i += 1
                    j = 0
                else:
                    j += 1
                if i >= map_size[0] - 2:
                    raise Exception("Incompatible map size and number of snakes")

            # Create the food always in the same spot
            food_location = [(map_size[0]-1, map_size[1]-1)]*max_turns_num*max_turns_num
            print(snake_locations)
            # Create the gym
            env = BattlesnakeGym(map_size=map_size, number_of_snakes=num_snakes,
                                 snake_spawn_locations=snake_locations,
                                 food_spawn_locations=food_location)

            actions =[[Snake.RIGHT]*num_snakes,
                 [Snake.DOWN]*num_snakes,
                 [Snake.LEFT]*num_snakes,
                 [Snake.UP]*num_snakes] * max_turns_num

            tic = time.time()
            _, _, _, info = simulate_snake(env, actions, render=False)
            toc = time.time()

            print("Map Size {}, Num Snake {}, Num Turns {}, Total time: {:.4f}s, Steps per seconds {:.4f}".format(
                map_size, num_snakes, info['current_turn'], toc-tic, info['current_turn']/(toc-tic)))


if __name__ == '__main__':
    map_sizes = [(8, 8), (10, 10), (12, 12), (14, 14), (20, 20)]
    number_of_snakes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    test_gym_performance(map_sizes, number_of_snakes)

