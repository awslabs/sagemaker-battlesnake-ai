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
import gym
from gym import spaces
from gym.utils import seeding
import json
import string

from .snake import Snakes
from .food import Food
from .game_state_parser import Game_state_parser
from .rewards import SimpleRewards
from .utils import get_random_coordinates, MultiAgentActionSpace, get_distance

class BattlesnakeGym(gym.Env):
    metadata = {
        "render.modes": ["human", "rgb_array", "ascii"],
        "observation.types": ["flat-num", "bordered-num",
                              "max-bordered-num",
                              "flat-51s", "bordered-51s", 
                              "max-bordered-51s"]
    }
    '''
    OpenAI Gym for BattlesnakeIO 
    Behaviour of snakes in Battlesnake.io based on https://docs.battlesnake.com/references/rules

    Parameters:
    ----------
    observation_type: str, options, options=["flat-num", "bordered-num",
                              "flat-51s", "bordered-51s"], default="flat-51s"
        Sets the observation space of the gym
        1- "flat-num" option will only provide food and snakes (represented by 1...length)
        2- "bordered-num" option will provide a border of -1 surrounding the gym
        3- "flat-51s" is the same as the flat option but the snake head is labeleld as 5 and the rest
           is labelled as 1
        4- "bordered-51s" similar to flat-51s
        5- "max-bordered-num" option will provide borders of -1 until a maximum map size of 21, 21
        6- "max-bordered-51s" similar to max-bordered-num
    
    map_size: (int, int), optional, default=(15, 15)
    
    number_of_snakes: int, optional, default=1

    snake_spawn_locations: [(int, int)] optional, default=[]
        Parameter to force snakes to spawn in certain positions. Used for testing
    
    food_spawn_location: [(int, int)] optional, default=[]
        Parameter to force food to spawn in certain positions. Used for testing
        Food will spawn in the coordinates provided in the list until the list is exhausted.
        After the list is exhausted, food will be randomly spawned

    verbose: Bool, optional, default=False

    initial_game_state: dict , default=None
        Dictionary to indicate the initial game state
        Dict is in the same form as in the battlesnake engine
        https://docs.battlesnake.com/references/api
    '''
    MAX_BORDER = (21, 21) # Largest map size (19, 19) + 2 for -1 borders
    def __init__(self, observation_type="flat-51s", map_size=(15, 15),
                 number_of_snakes=4, 
                 snake_spawn_locations=[], food_spawn_locations=[],
                 verbose=False, initial_game_state=None, rewards=SimpleRewards()):
        
        self.map_size = map_size
        self.number_of_snakes = number_of_snakes
        self.initial_game_state = initial_game_state
        self.snake_spawn_locations = snake_spawn_locations
        self.food_spawn_locations = food_spawn_locations
        
        self.number_of_snakes = number_of_snakes
        self.map_size = map_size

        self.action_space = MultiAgentActionSpace(
            [spaces.Discrete(4) for _ in range(number_of_snakes)])

        self.observation_type = observation_type
        self.observation_space = self.get_observation_space()
        
        self.viewer = None
        self.state = None
        self.verbose = verbose
        self.rewards = rewards

    def get_observation_space(self):
        '''
        Helper function to define the observation space given self.map_size, self.number_of_snakes
        and self.observation_type
        '''
        if "flat" in self.observation_type:
            observation_space = spaces.Box(low=-1, high=5,
                                           shape=(self.map_size[0],
                                                  self.map_size[1],
                                                  self.number_of_snakes+1),
                                           dtype=np.uint8)
        elif "bordered" in self.observation_type:
            if "max-bordered" in self.observation_type:
                border_size = self.MAX_BORDER[0] - self.map_size[0]
            else:
                border_size = 2
            observation_space = spaces.Box(low=-1, high=5,
                                           shape=(self.map_size[0]+border_size,
                                                  self.map_size[1]+border_size,
                                                  self.number_of_snakes+1),
                                           dtype=np.uint8)
        return observation_space

    def initialise_game_state(self, game_state_dict):
        '''
        Function to initialise the gym with outputs of env.render(mode="ascii")
        The output is fed in through a text file containing the rendered ascii string
        '''
        gsp = Game_state_parser(game_state_dict)
        # Check that the current map size and number of snakes are identical to the states
                
        assert np.array_equal(gsp.map_size, self.map_size), "Map size of the game state is incorrect"
        assert gsp.number_of_snakes == self.number_of_snakes, "Number of names of the game state is incorrect"

        return gsp.parse()
        
    def seed(self, seed):
        '''
        Inherited function of the openAI gym to set the randomisation seed.
        '''
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, map_size=None):
        '''
        Inherited function of the openAI gym to reset the environment.

        Parameters:
        -----------
        map_size: (int, int), default None
            Optional paramter to reset the map size
        '''
        if map_size is not None:
            self.observation_space = self.get_observation_space()
            self.map_size = map_size
        
        if self.initial_game_state is not None:
            self.snakes, self.food, self.turn_count = self.initialise_game_state(self.initial_game_state)
        else:
            self.turn_count = 0

            self.snakes = Snakes(self.map_size, self.number_of_snakes, self.snake_spawn_locations)
            self.food = Food(self.map_size, self.food_spawn_locations)
            self.food.spawn_food(self.snakes.get_snake_51_map())

        dones = {i:False for i in range(self.number_of_snakes)}
        
        snakes_health = {}
        snake_info = {}
        self.snake_max_len = {}
        for i, snake in enumerate(self.snakes.get_snakes()):
            snakes_health[i] = snake.health
            snake_info[i] = "Did not collide" 
            self.snake_max_len[i] = 0
        info = {'current_turn': self.turn_count,
                'snake_health': snakes_health,
                'snake_info': snake_info, 
                'snake_max_len': self.snake_max_len}
        return self._get_observation(), {}, dones, info

    def _did_snake_collide(self, snake, snakes_to_be_killed):
        '''
        Helper function to check if a snake has collided into something else. Checks the following:
        1) If the snake's head hit a wall (i.e., if the head is outside of the map)
        2) Check if the snake collided with another snake's head (entering the same tile and adjacent)
        3) Check if the snake ran into another snake's body (itself and other snakes)
        4) Check if the snake's body hit another snake's head
        
        Parameter:
        ----------
        snake: Snake
        
        snakes_to_be_killed: a list of snakes that will be killed in the end of the turn.

        Returns:
        ----------
        should_kill_snake: Bool
            Boolean to indicate if the snake is dead or not

        collision_outcome: options = ["Snake hit wall", 
                                      "Snake was eaten - same tile", 
                                      "Snake was eaten - adjacent tile", 
                                      "Snake hit body - hit itself", 
                                      "Snake hit body - hit other", 
                                      "Did not collide",
                                      "Ate another snake",
                                      "Other snake hit body"]
        '''       
        snake_head_location = snake.get_head()
        ate_another_snake = False
        snakes_eaten_this_turn = []
        
        # 1) Check if the snake ran into a wall
        outcome = "Snake hit wall"
        if snake.is_head_outside_map():
            if self.verbose: print(outcome)
            should_kill_snake = True
            return should_kill_snake, outcome

        # 2.1) Check if snake's head collided with another snake's head when they both entered the same tile
        # e.g.,:
        # 
        #  | |< S1   
        #   ^ 
        #   S2
        for other_snake in self.snakes.get_snakes():
            if other_snake == snake:
                continue
            if other_snake.is_alive():
                other_snake_head = other_snake.get_head()
                if np.array_equal(snake_head_location, other_snake_head):
                    if other_snake.get_size() >= snake.get_size():
                        outcome = "Snake was eaten - same tile"
                        if self.verbose: print(outcome)
                        return True, outcome
                    else:
                        ate_another_snake = True
                        snakes_eaten_this_turn.append(other_snake)
                
        # 2.2) Check if snake's head collided with another snakes head when they were adjacent to one another
        # (i.e., that the heads swapped positions)
        #
        #    S1     S1
        #   |  |> <|  |
        #
        for other_snake in self.snakes.get_snakes():
            if other_snake == snake:
                continue
            # Check if snake swapped places with the other_snake.
            # 1) check if heads are adjacent
            # 2) check if heads swapped places
            if other_snake.is_alive():
                other_snake_head = other_snake.get_head()
                if get_distance(snake_head_location, other_snake_head) == 1:
                    if np.array_equal(snake_head_location, other_snake.get_previous_snake_head())\
                       and np.array_equal(other_snake_head, snake.get_previous_snake_head()):
                        if other_snake.get_size() >= snake.get_size():
                            outcome = "Snake was eaten - adjacent tile"
                            if self.verbose: print(outcome)
                            return True, outcome
                        else:
                            ate_another_snake = True
                            snakes_eaten_this_turn.append(other_snake)

        # 3.1) Check if snake ran into it's own body
        outcome = "Snake hit body - hit itself"
        for self_body_locations in snake.get_body():
            if np.array_equal(snake_head_location, self_body_locations):
                if self.verbose: print("Snake hit itself")
                return True, outcome
            
        # 3.2) Check if snake ran into another snake's body
        outcome = "Snake hit body - hit other"
        snake_binary_map = self.snakes.get_snake_51_map(excluded_snakes=[snake]+snakes_eaten_this_turn)
        if snake_binary_map[snake_head_location[0], snake_head_location[1]] == 1:
            if self.verbose: print("Snake hit another snake")
            return True, outcome

        # 4) Check if another snake ran into this snake
        for other_snake in self.snakes.get_snakes():
            if other_snake == snake:
                continue
            if other_snake.is_alive():
                if other_snake not in snakes_to_be_killed:
                    other_snake_head = other_snake.get_head()
                    for location in snake.get_body():
                        if np.array_equal(location, other_snake_head):
                            return False, "Other snake hit body"
        
        if ate_another_snake:                            
            return False, "Ate another snake"

        return False, "Did not collide"

    def step(self, actions, episodes=None):
        '''
        Inherited function of the openAI gym. The steps taken mimic the steps provided in 
        https://docs.battlesnake.com/references/rules -> Programming Your Snake -> 3) Turn resolution.
        
        Parameters:
        ---------
        action: np.array(number_of_snakes)
            Array of integers containing an action for each number of snake. 
            The integers range from 0 to 3 corresponding to Snake.UP, Snake.DOWN, Snake.LEFT, Snake.RIGHT 
            respectively

        Returns:
        -------

        observation: np.array
            Output of the current state of the gym

        reward: {}
            The rewards obtained by each snake. 
            Dictionary is of length number_of_snakes

        done: Bool
            Indication of whether the gym is complete or not.
            Gym is complete when there is only 1 snake remaining
        '''

        # setup reward dict
        reward = {}
        snake_info = {}

        # DEBUGING
        json_before_moving = self.get_json()
        
        # Reduce health and move
        for i, snake in enumerate(self.snakes.get_snakes()):
            reward[i] = 0
            if not snake.is_alive():
                continue

            # Reduce health by one
            snake.health -= 1
            if snake.health == 0:
                snake.kill_snake()
                reward[i] += self.rewards.get_reward("starved", i, episodes)
                snake_info[i] = "Starved"
                continue

            action = actions[i] 
            is_forbidden = snake.move(action)
            if is_forbidden:
                snake.kill_snake()
                reward[i] += self.rewards.get_reward("forbidden_move", i, episodes)
                snake_info[i] = "Forbidden move"
            
        # check for food and collision
        number_of_food_eaten = 0
        number_of_snakes_alive = 0

        
        # DEBUGING
        json_after_moving = self.get_json()
        
        snakes_to_be_killed = []
        for i, snake in enumerate(self.snakes.get_snakes()):
            if not snake.is_alive():
                continue

            snake_head_location = snake.get_head()

            # Check for collisions with the snake
            should_kill_snake, outcome = self._did_snake_collide(snake, snakes_to_be_killed)
            if should_kill_snake:
                snakes_to_be_killed.append(snake)
            snake_info[i] = outcome

            # Check if snakes ate any food
            if not should_kill_snake and self.food.does_coord_have_food(snake_head_location):
                number_of_food_eaten += 1
                snake.set_ate_food()
                self.food.remove_food_from_coord(snake_head_location)
                reward[i] += self.rewards.get_reward("ate_food", i, episodes)

            # Calculate rewards for collision
            if outcome == "Snake hit wall":
                reward[i] += self.rewards.get_reward("hit_wall", i, episodes)
                
            elif outcome == "Snake was eaten - same tile":
                reward[i] += self.rewards.get_reward("was_eaten", i, episodes)
                
            elif outcome == "Snake was eaten - adjacent tile":
                reward[i] += self.rewards.get_reward("was_eaten", i, episodes)
                
            elif outcome == "Snake hit body - hit itself":
                reward[i] += self.rewards.get_reward("hit_self", i, episodes)
                
            elif outcome == "Snake hit body - hit other":
                reward[i] += self.rewards.get_reward("hit_other_snake", i,
                                                    episodes)

            elif outcome == "Other snake hit body":
                reward[i] += self.rewards.get_reward("other_snake_hit_body", i,
                                                    episodes)
                
            elif outcome == "Did not collide":
                pass
                
            elif outcome == "Ate another snake":
                reward[i] += self.rewards.get_reward("ate_another_snake", i,
                                                    episodes)
        for snake_to_be_killed in snakes_to_be_killed:
            snake_to_be_killed.kill_snake()
        
        snakes_alive = []
        for i, snake in enumerate(self.snakes.get_snakes()):
            snakes_alive.append(snake.is_alive())
            if snake.is_alive():
                number_of_snakes_alive += 1
                reward[i] += self.rewards.get_reward("another_turn", i, episodes)
        
        self.food.end_of_turn(self.snakes.get_snake_51_map())

        if self.number_of_snakes > 1 and np.sum(snakes_alive) <= 1:
            done = True
            for i, is_snake_alive in enumerate(snakes_alive):
                if is_snake_alive:
                    reward[i] += self.rewards.get_reward("won", i, episodes)
                else:
                    reward[i] += self.rewards.get_reward("died", i, episodes)
        else:
            done = False
            
        snake_alive_dict = {i: a for i, a in enumerate(np.logical_not(snakes_alive).tolist())}
        self.turn_count += 1

        snakes_health = {}
        for i, snake in enumerate(self.snakes.get_snakes()):
            snakes_health[i] = snake.health
            if snake.is_alive():
                self.snake_max_len[i] += 1
            if i not in snake_info:
                snake_info[i] = "Dead"
                
        sum_map = self.snakes.get_snake_51_map()
        if np.max(sum_map) > 5 or 2 in sum_map:
            print("snake info {}".format(snake_info))
            print("actions {}".format(actions))
            print("before moving json {}".format(json_before_moving))
            print("after moving json {}".format(json_after_moving))
            print("final json {}".format(self.get_json()))
            raise
            
        return self._get_observation(), reward, snake_alive_dict, {'current_turn': self.turn_count,
                                                                   'snake_health': snakes_health,
                                                                   'snake_info': snake_info,
                                                                   'snake_max_len': self.snake_max_len}
                
    def _get_observation(self):
        '''
        Helper function to generate the output observation.
        '''
        if "flat" in self.observation_type:
            return self._get_state()
        elif "bordered" in self.observation_type:
            state = self._get_state()

            if "max-bordered" in self.observation_type:
                border_size = self.MAX_BORDER[0] - self.map_size[0]
            else: 
                border_size = 2
                
            bordered_state_shape = (state.shape[0]+border_size, state.shape[1]+border_size,
                                    state.shape[2])
            bordered_state = np.ones(shape=bordered_state_shape)*-1
            
            b = int(border_size/2)
            bordered_state[b:-b, b:-b,:] = state
            return bordered_state

    def _get_state(self):
        ''''
        Helper function to generate the state of the game.

        Returns:
        --------
        state: np.array(map_size[1], map_size[2], number_of_snakes + 1)
            state[:, :, 0] corresponds to a binary image of the location of the food
            state[:, :, 1:] corrsponds to binary images of the locations of other snakes
        '''
        FOOD_INDEX = 0
        SNAKE_INDEXES = FOOD_INDEX + np.array(range(1, self.number_of_snakes + 1))

        depth_of_state = 1 + self.snakes.number_of_snakes
        state = np.zeros((self.map_size[0], self.map_size[1], depth_of_state),
                         dtype=np.uint8)

        # Include the postions of the food
        state[:, :, FOOD_INDEX] = self.food.get_food_map()
        
        # Include the positions of the snakes
        if "51s" in self.observation_type:
            state[:, :, SNAKE_INDEXES] = self.snakes.get_snake_depth_51_map()
        else:
            state[:, :, SNAKE_INDEXES] = self.snakes.get_snake_depth_numbered_map()   
        return state

    def _get_board(self, state):
        ''''
        Generate visualisation of the gym. Based on the state (generated by _get_state).
        '''
        
        FOOD_INDEX = 0
        SNAKE_INDEXES = FOOD_INDEX + np.array(range(1, self.number_of_snakes + 1))

        BOUNDARY = 20
        BOX_SIZE = 40
        SPACE_BETWEEN_BOXES = 10
        snake_colours = self.snakes.get_snake_colours()
        
        # Create board
        board_size = (self.map_size[0]*(BOX_SIZE + SPACE_BETWEEN_BOXES) + 2*BOUNDARY,
                      self.map_size[1]*(BOX_SIZE + SPACE_BETWEEN_BOXES) + 2*BOUNDARY)
        board = np.ones((board_size[0], board_size[1], 3), dtype=np.uint8) * 255

        # Create boxes
        for i in range(0, self.map_size[0]):
            for j in range(0, self.map_size[1]):
                state_value = state[i][j]

                t_i1 = BOUNDARY + i * (BOX_SIZE + SPACE_BETWEEN_BOXES)
                t_i2 = t_i1 + BOX_SIZE

                t_j1 = BOUNDARY + j * (BOX_SIZE + SPACE_BETWEEN_BOXES)
                t_j2 = t_j1 + BOX_SIZE


                board[t_i1:t_i2, t_j1:t_j2] = 255 * 0.7
                
                # If state contains food
                if state_value[FOOD_INDEX] >= 1:
                    box_margin = int(BOX_SIZE/5)
                    board[t_i1 + box_margin:t_i2 - box_margin,
                          t_j1 + box_margin:t_j2 - box_margin] = [255, 0, 0]

                # If state contains a snake body
                if 1 in state_value[SNAKE_INDEXES]:
                    snake_present_in = np.argmax(state_value[SNAKE_INDEXES])
                    board[t_i1:t_i2, t_j1:t_j2] = snake_colours[snake_present_in]
                    
                # If state contains a snake head
                if 5 in state_value[SNAKE_INDEXES]:
                    snake_present_in = np.argmax(state_value[SNAKE_INDEXES])
                    board[t_i1:t_i2, t_j1:t_j2] = snake_colours[snake_present_in]
                    t_i_h = 10
                    t_j_h = 10
                    board[(t_i1+t_i_h):(t_i2-t_i_h), (t_j1+t_j_h):(t_j2-t_j_h)] = [255, 255, 255]

        return board
    
    def get_json(self):
        '''
        Generate a json representation of the gym following the same input as the battlesnake
        engine.
        
        Return:
        -------
        json: {}
            Json in the same representation of board.
        '''
        json = {}
        json["turn"] = self.turn_count
        
        # Get food
        food_list = []
        y, x = np.where(self.food.locations_map==1)
        for x_, y_ in zip(x, y):
            food_list.append({"x": x_, "y": y_})
        
        # Get snakes
        snake_dict_list = []
        for i, snakes in enumerate(self.snakes.snakes):
            snake_location = []
            for coord in snakes.locations[::-1]:
                snake_location.append({"x": coord[1], "y": coord[0]})
                
            snake_dict = {}
            snake_dict["health"] = snakes.health
            snake_dict["body"] = snake_location
            snake_dict["id"] = i
            snake_dict["name"] = "Snake {}".format(i)
            snake_dict_list.append(snake_dict)

        json["board"] = {"height": self.map_size[0],
                         "width": self.map_size[1],
                         "food": food_list,
                         "snakes": snake_dict_list 
                        }
        return json

    def _get_ascii(self):
        '''
        Generate visualisation of the gym. Prints ascii representation of the gym.
        Could be used as an input to initialise the gym.

        Return:
        -------
        ascii_string: str
            String visually depicting the gym
             - The walls of the gym are labelled with "*"
             - Food is labelled with "@"
             - Snakes characters (head is the uppercase letter)
             - The health of each snake is on the bottom of the gym
        '''
        FOOD_INDEX = 0
        SNAKE_INDEXES = FOOD_INDEX + np.array(range(1, self.number_of_snakes + 1))

        ascii_array = np.empty(shape=(self.map_size[0] + 2, self.map_size[1] + 2), dtype="object")

        # Set the borders of the image
        ascii_array[0, :] = "*"
        ascii_array[-1, :] = "*"
        ascii_array[:, 0] = "*"
        ascii_array[:, -1] = "*"

        # Populate food
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                if self.food.locations_map[i, j]:
                    ascii_array[i+1, j+1] = " @"

        # Populate snakes
        for idx, snake in enumerate(self.snakes.get_snakes()):
            snake_character = string.ascii_lowercase[idx]
            for snake_idx, location in enumerate(snake.locations):
                snake_idx = len(snake.locations) - snake_idx - 1
                ascii_array[location[0]+1, location[1]+1] = snake_character + str(snake_idx)

        # convert to string
        ascii_string = ""
        for i in range(ascii_array.shape[0]):
            for j in range(ascii_array.shape[1]):
                if j == 0 and i == 0:
                    ascii_string += "* "
                elif j == 0 and i == ascii_array.shape[0] - 1:
                    ascii_string += "* "
                elif j == 0:
                    ascii_string += "*|"
                elif ascii_array[i][j] == "*":
                    ascii_string += "\t - *"
                elif ascii_array[i][j] is None:
                    ascii_string += "\t . |"
                else:
                    ascii_string += ascii_array[i][j] + "\t . |"
            ascii_string += "\n"

        # Print turn count
        ascii_string += "Turn = {}".format(self.turn_count) + "\n"
        
        # Print snake health
        for idx, snake in enumerate(self.snakes.get_snakes()):
            ascii_string += "{} = {}".format(string.ascii_lowercase[idx], snake.health) + "\n"
            
        return ascii_string
        
    def render(self, mode="human"):
        '''
        Inherited function from openAI gym to visualise the progression of the gym
        
        Parameter:
        ---------
        mode: str, options=["human", "rgb_array"]
            mode == human will present the gym in a separate window
            mode == rgb_array will return the gym in np.arrays
        '''
        state = self._get_state()
        if mode == "rgb_array":
            return self._get_board(state)
        elif mode == "ascii":
            ascii = self._get_ascii()
            print(ascii)
            # for _ in range(ascii.count('\n')):
            #     print("\033[A")
            return ascii
        elif mode == "human":
            from gym.envs.classic_control import rendering
            board = self._get_board(state)
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(board)
            return self.viewer.isopen
