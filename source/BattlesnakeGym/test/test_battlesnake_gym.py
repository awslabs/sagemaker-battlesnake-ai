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
import unittest

import numpy as np

from battlesnake_gym.snake_gym import BattlesnakeGym
from battlesnake_gym.snake import Snake

from .test_utils import grow_snake, grow_two_snakes, should_render, simulate_snake

class TestBattlesnakeGym(unittest.TestCase):
    '''
    Test the behaviours of the BattlesnakeGym including:
    - Test spawning 
    - Test deterministic spawning (spawning food and snakes in certain locations for testing)
    - Test that the snakes moves correctly
    - Test the snake health and that the snake dies after it's health is 0
    - Test the food eating mechanism
    - Test that the snake dies after hitting a wall
    - Test that the snake dies after itself
    - Test that snake dies after other snake
    - Test that the snake eats another when moving into the same tile 
    - Test that the snake eats another when moving into adjacent tiles
    - Test that the both snakes (of the same size) die when eating each other
    - Test states returned
    - Test food spawning
    
    The output of the program should be:
        Snake hit itself
        Snake hit wall
        .Snake was eaten - adjacent tile
        .Snake was eaten - adjacent tile
        Snake was eaten - adjacent tile
        Snake was eaten - same tile
        Snake hit another snake
    '''

    def test_snake_eaten_adjacent_tile_same_size(self):
        '''
        Tests that if two snakes of the same size eat each other, they both die
        see: outcome option = "Snake was eaten - adjacent tile" in snake_gym._did_snake_collide
        '''
        snake_location = [(4, 1), (4, 12)]
        food_location = [(4, 2), (4, 9), (4, 3), (4, 8), 
                         (0, 0), (0, 0), (0, 0)]
        env = BattlesnakeGym(map_size=(13, 13), number_of_snakes=2,
                             snake_spawn_locations=snake_location,
                             food_spawn_locations=food_location,
                             verbose=True)

        env.food.max_turns_to_next_food_spawn = 2  # Hack to make sure that food is spawned every turn
        
        actions = [[Snake.RIGHT, Snake.LEFT], [Snake.RIGHT, Snake.LEFT], [Snake.RIGHT, Snake.LEFT],
                   [Snake.RIGHT, Snake.LEFT], [Snake.RIGHT, Snake.LEFT], [Snake.RIGHT, Snake.LEFT],
                   [Snake.RIGHT, Snake.LEFT]]

        simulate_snake(env, actions, render=should_render(), break_with_done=False)

        snakes_alive = [snake.is_alive() for snake in env.snakes.get_snakes()]
        self.assertTrue(np.sum(snakes_alive) == 0)

        env.close()

    def test_snake_eaten_adjacent_tile(self):
        '''
        Tests that the snake dies if it's eaten by a bigger snake
        see: outcome option = "Snake was eaten - adjacent tile" in snake_gym._did_snake_collide
        '''
        env = grow_two_snakes(snake_starting_positions=[(0, 0), (5, 0)])
        actions_snake1 = [[Snake.DOWN], [Snake.LEFT]] + [[Snake.UP]]*4
        
        actions_snake2 = [[Snake.UP]]*3 + [[Snake.RIGHT]]*3
        tmp_actions = list(zip(actions_snake1, actions_snake2))
        actions = []
        for action in tmp_actions:
            actions.append(np.array([action[0], action[1]]))
        simulate_snake(env, actions, render=should_render(), break_with_done=False)

        snakes_alive = [snake.is_alive() for snake in env.snakes.get_snakes()]
        self.assertTrue(np.sum(snakes_alive) == 2)

        actions = [np.array([Snake.LEFT, Snake.RIGHT])]

        simulate_snake(env, actions, render=should_render(), break_with_done=False)
        snakes_alive = [snake.is_alive() for snake in env.snakes.get_snakes()]
        self.assertTrue(snakes_alive == [True, False])

        env.close()

    def test_snake_eaten_same_tile(self):
        '''
        Tests that the snake dies if it's eaten by a bigger snake
        see: outcome option = "Snake was eaten - same tile" in snake_gym._did_snake_collide
        '''
        env = grow_two_snakes(snake_starting_positions=[(0, 0), (5, 1)])
        actions_snake1 = [[Snake.DOWN], [Snake.LEFT]] + [[Snake.UP]]*3
        
        actions_snake2 = [[Snake.UP]]*3 + [[Snake.RIGHT]]*2
        tmp_actions = list(zip(actions_snake1, actions_snake2))
        actions = []
        for action in tmp_actions:
            actions.append(np.array([action[0], action[1]]))
        simulate_snake(env, actions, render=should_render(), break_with_done=False)

        snakes_alive = [snake.is_alive() for snake in env.snakes.get_snakes()]
        self.assertTrue(np.sum(snakes_alive) == 2)

        actions = [np.array([Snake.UP, Snake.RIGHT])]

        simulate_snake(env, actions, render=should_render(), break_with_done=False)
        snakes_alive = [snake.is_alive() for snake in env.snakes.get_snakes()]

        self.assertTrue(snakes_alive == [True, False])

        env.close()

    def test_snake_hit_other_snake(self):
        '''
        Tests that the snake dies after it hits the body of another snake
        see: outcome option = "Snake hit body - hit other" in snake_gym._did_snake_collide
        '''
        env = grow_two_snakes(snake_starting_positions=[(0, 0), (5, 0)])
        actions_snake1 = [[Snake.DOWN], [Snake.LEFT], [Snake.UP], [Snake.UP],
                          [Snake.LEFT], [Snake.LEFT]]
        
        actions_snake2 = [[Snake.RIGHT]]*2 + [[Snake.UP]]*3
        tmp_actions = list(zip(actions_snake1, actions_snake2))
        actions = []
        for action in tmp_actions:
            actions.append(np.array([action[0], action[1]]))
        simulate_snake(env, actions, render=should_render(), break_with_done=False)

        snakes_alive = [snake.is_alive() for snake in env.snakes.get_snakes()]
        self.assertTrue(np.sum(snakes_alive) == 2)
        
        actions = [np.array([Snake.LEFT, Snake.UP])]

        simulate_snake(env, actions, render=should_render(), break_with_done=False)
        snakes_alive = [snake.is_alive() for snake in env.snakes.get_snakes()]
        self.assertTrue(snakes_alive == [False, True])

        env.close()

    def test_snake_die_when_hit_self(self):
        '''
        Tests that the snake dies after it hits the body of itself
        see: outcome option = "Snake hit body - hit itself" in snake_gym._did_snake_collide
        '''

        env = grow_snake()
        actions = [[Snake.DOWN]]
        actions += [[Snake.LEFT], [Snake.UP], [Snake.RIGHT]]

        simulate_snake(env, actions, render=should_render(), break_with_done=False)
        
        # Check snake died
        self.assertTrue(np.sum(env.snakes.get_snake_51_map()) == 0)
        env.close()
    
    def test_snake_die_when_hit_wall(self):
        '''
        Tests that the snake dies after it hits a wall
        see: outcome option = "Snake hit wall" in snake_gym._did_snake_collide
        '''

        env = grow_snake()

        actions = [[Snake.DOWN]]
        actions += [[Snake.LEFT]] * 10
        simulate_snake(env, actions, render=should_render(), break_with_done=False)

        # Check snake died
        self.assertTrue(np.sum(env.snakes.get_snake_51_map()) == 0)
        env.close()

    def test_snake_eating(self):
        '''
        Test that snake grows after eating
        '''
        env = grow_snake()

        self.assertTrue(np.sum(env.snakes.get_snake_51_map()) == 12)
        env.close()

    def test_snake_health(self):
        '''
        Test that snake dies after moving 100 times. i.e., health == 0
        '''

        snake_location = [(0, 0)]
        food_location = [(5, 5) for _ in range(0, 200)]
        env = BattlesnakeGym(map_size=(9, 10), number_of_snakes=1,
                          snake_spawn_locations=snake_location,
                          food_spawn_locations=food_location,
                          verbose=True)

        actions = [[Snake.RIGHT]]
        simulate_snake(env, actions, render=should_render())
        self.assertTrue(env.snakes.get_snakes()[0].health == Snake.FULL_HEALTH - 1)

        actions = []
        for i in range(1, Snake.FULL_HEALTH - 1):
            if int(i % 32 / 8) == 0:
                actions.append([Snake.RIGHT])
                continue
            if int(i % 32 / 16) == 0:
                actions.append([Snake.DOWN])
                continue
            if int(i % 32 / 24) == 0:
                actions.append([Snake.LEFT])
                continue
            if int(i % 32 / 32) == 0:
                actions.append([Snake.UP])
                continue
        simulate_snake(env, actions, render=should_render(), break_with_done=False)
        self.assertTrue(env.snakes.get_snakes()[0].health == 1)
        self.assertTrue(env.snakes.get_snakes()[0].is_alive())
        
        actions = [[Snake.RIGHT]]
        simulate_snake(env, actions, render=should_render())
        self.assertTrue(env.snakes.get_snakes()[0].health == 0)
        self.assertFalse(env.snakes.get_snakes()[0].is_alive())

        # Check snake died
        actions = [[Snake.RIGHT]]
        simulate_snake(env, actions, render=should_render())
        self.assertTrue(np.sum(env.snakes.get_snake_51_map()) == 0)
        env.close()

    def test_snake_move(self):
        '''
        Test that the snake moves correctly.
        Spawn a snake in a certain location and moving them one space in each direction 
        UP, DOWN, LEFT, RIGHT
        '''
        snake_location = [(4, 4)]
        food_location = [(5, 5)]
        env = BattlesnakeGym(map_size=(9, 10), number_of_snakes=1,
                          snake_spawn_locations=snake_location,
                          food_spawn_locations=food_location)

        actions = [[Snake.UP]]
        simulate_snake(env, actions, render=should_render())
        snake_location_moved = [(4, 4), (3, 4)]
        self.assertTrue(
            np.array_equal(env.snakes.get_snakes()[0].locations, snake_location_moved))

        actions = [[Snake.LEFT]]
        simulate_snake(env, actions, render=should_render())
        snake_location_moved = [(4, 4), (3, 4), (3, 3)]
        self.assertTrue(
            np.array_equal(env.snakes.get_snakes()[0].locations, snake_location_moved))

        actions = [[Snake.DOWN]]
        simulate_snake(env, actions, render=should_render())
        snake_location_moved = [(3, 4), (3, 3), (4, 3)]
        self.assertTrue(
            np.array_equal(env.snakes.get_snakes()[0].locations, snake_location_moved))
        
        actions = [[Snake.RIGHT]]
        simulate_snake(env, actions, render=should_render())
        snake_location_moved = [(3, 3), (4, 3), (4, 4)]
        self.assertTrue(
            np.array_equal(env.snakes.get_snakes()[0].locations, snake_location_moved))
        env.close()
    
    def test_random_spawning(self):
        '''
        Test that snakes and food are correct when randomly spawned
        '''
        env = BattlesnakeGym(map_size=(9, 9), number_of_snakes=1)

        # Check that a snake is spawned on the board        
        self.assertTrue(len(env.snakes.snakes) > 0)

        # Check that there is a food on the board
        self.assertTrue(env.food.locations_map.sum() > 0)
        env.close()

    def test_spawning(self):
        '''
        Test that snakes and food are correct when deterministically spawned (for testing)
        '''
        snake_location = [(4, 4)]
        food_location = [(5, 5)]
        env = BattlesnakeGym(map_size=(9, 9), number_of_snakes=1,
                          snake_spawn_locations=snake_location,
                          food_spawn_locations=food_location)

        # Check that the snake is spawned correctly
        self.assertTrue(np.array_equal(env.snakes.snakes[0].locations,  snake_location))

        # Check that food is spanwed correctly
        self.assertTrue(env.food.locations_map[food_location[0][0], food_location[0][1]] == 1)
        env.close()

    def test_states(self):
        '''
        Test that the state returned is correct
        '''
        snake_location = [(0, 0)]
        food_location = [(1, 0), (1, 2), (2, 0), (2, 0), (2, 0)]
        env = BattlesnakeGym(map_size=(3, 3), number_of_snakes=1,
                          snake_spawn_locations=snake_location,
                          food_spawn_locations=food_location)
        env.food.max_turns_to_next_food_spawn = 2 #Hack to make sure that food is spawned every turn
        
        actions = [[Snake.DOWN], [Snake.RIGHT], [Snake.RIGHT], [Snake.DOWN], [Snake.LEFT]]
        observation, _, _, _ = simulate_snake(env, actions, render=False, break_with_done=False)

        food_state = np.zeros(shape=(3, 3), dtype=np.uint8)
        food_state[1, 2] = 1

        snake_state = np.zeros(shape=(3, 3), dtype=np.uint8)
        snake_state[1, 1] = 1
        snake_state[2, 2] = 1
        snake_state[1, 2] = 1
        snake_state[2, 1] = 5

        self.assertTrue(np.array_equal(observation[:, :, 0],  food_state))
        self.assertTrue(np.array_equal(observation[:, :, 1],  snake_state))

if __name__ == '__main__':
    unittest.main()
    
