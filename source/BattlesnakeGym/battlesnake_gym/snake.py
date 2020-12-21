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

from .utils import get_random_coordinates

class Snake:
    '''
    The Snake class mimics the behaviour of snakes in Battlesnake.io based on 
    https://docs.battlesnake.com/rules
    
    Parameters:
    -----------
    starting_position: (int, int)
        The initial position of the snake

    map_size: (int, int)
        The size of the map

    '''

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    FULL_HEALTH = 100
    
    def __init__(self, starting_position, map_size):
        self.health = self.FULL_HEALTH
        self.locations = [] # Head of the snake is element n and the end is element 0
        self.locations.append(starting_position)
        self.facing_direction = None
        self._is_alive = True
        self.ate_food = False
        self.map_size = map_size
        self.colour = list(np.random.choice(range(256), size=3))
        self._number_of_initial_body_stacking = 2 # At the start of the game, snakes of size 3 are stacked.
        # self._number_of_initial_body_stacking == 2 to account for the initial body

    @classmethod
    def make_from_list(cls, locations, health, map_size):
        '''
        Class method to make a snake from a list of coordinates.
        Parameters:
        ----------
        locations: [(int, int)]
            An ordered list of coordinates of the body (y, x)
        health: int
            The health of the snake
        map_size: (int, int)
        '''
        tmp_locations = []
        for i, j in locations[::-1]: # head is element n
            tmp_locations.append(np.array([i, j])) 

        if len(tmp_locations) == 0:
            head = None
        else:
            head = tmp_locations[-1]
        cls = Snake(head, map_size)
        cls.locations = tmp_locations
        cls.health = health
        if len(tmp_locations) == 0:
            cls.kill_snake()

        if len(cls.locations) > 1:
            # Calculate the facing direction with the head and the next location
            snake_head = cls.locations[-1]
            snake_2nd_body = cls.locations[-2]
            difference = (snake_head[0] - snake_2nd_body[0], snake_head[1] - snake_2nd_body[1])
            if difference[0] == -1 and difference[1] == 0:
                cls.facing_direction = Snake.UP
            elif difference[0] == 1 and difference[1] == 0:
                cls.facing_direction = Snake.DOWN
            elif difference[0] == 0 and difference[1] == -1:
                cls.facing_direction = Snake.LEFT
            elif difference[0] == 0 and difference[1] == 1:
                cls.facing_direction = Snake.RIGHT
        return cls
        
    def move(self, direction):
        '''
        Moves the snakes in the direction stated

        If the direction 
        
        Parameters:
        -----------
        direction: int, options: [Snake.UP, Snake.DOWN, Snake.LEFT or Snake.RIGHT]
            Move the start towards the directions

        Returns:
        -------
        is_forbidden: Boolean
            Whether the move was a forbidden one: moving backward in its own body
        '''
        is_forbidden = False
        if not self._is_alive:
            return is_forbidden
        
        if self.facing_direction == None:
            self.facing_direction == direction

        if self.is_facing_opposite_of_direction(direction) and len(self.locations) > 0:
            direction = self.facing_direction
            is_forbidden = True

        head = self.get_head()
        new_head = self._translate_coordinate_in_direction(head, direction)

        # If the snake is within the first 3 turns of being alive, do no remove the end
        if self._number_of_initial_body_stacking > 0:
            self._number_of_initial_body_stacking -= 1
        # If the snake ate food, do not remove the end
        elif self.ate_food:
            self.ate_food = False
        else:
            self.locations = self.locations[1:] # remove the end
        self.locations.append(new_head)
        self.facing_direction = direction
        return is_forbidden
        
    def is_facing_opposite_of_direction(self, direction):
        '''
        Function to indicate if the indended direction is in the opposite of 
        the direction in which is snake is travelling

        Parameters:
        -----------
        direction: int, options: [Snake.UP, Snake.DOWN, Snake.LEFT or Snake.RIGHT]
            Direction intended for the snake to travel
        '''
        if self.facing_direction == self.UP and direction == self.DOWN:
            return True
        if self.facing_direction == self.DOWN and direction == self.UP:
            return True
        if self.facing_direction == self.RIGHT and direction == self.LEFT:
            return True
        if self.facing_direction == self.LEFT and direction == self.RIGHT:
            return True
        return False

    def get_previous_snake_head(self):
        '''
        Returns the location of head in the previous time step
        
        Move 1 space in the opposite direction of self.facing direction
        '''
        head = self.get_head()
        previous_head = np.copy(head)
        
        if self.facing_direction == Snake.UP:
            previous_head[0] += 1
        elif self.facing_direction == Snake.DOWN:
            previous_head[0] -= 1
        elif self.facing_direction == Snake.RIGHT:
            previous_head[1] -= 1
        elif self.facing_direction == Snake.LEFT:
            previous_head[1] += 1
        return previous_head

    def get_head(self):
        return self.locations[-1]

    def get_tail(self):
        return self.locations[0]

    def get_body(self):
        return self.locations[:-1]

    def _translate_coordinate_in_direction(self, origin, direction):
        '''
        Helper function to translate a coordinate to a direction
        
        Parameters:
        -----------
        origin: (int, int)
            Coordinate to be moved

        Direction: int, options: [Snake.UP, Snake.DOWN, Snake.LEFT or Snake.RIGHT
            Direction to be moved

        Returns:
        -------
        coordinate: (int, int)
            Translated coordinate
        '''
        new_coordinate = np.copy(origin)
        if direction == self.UP:
            new_coordinate[0] = new_coordinate[0] - 1
        elif direction == self.DOWN:
            new_coordinate[0] = new_coordinate[0] + 1
        elif direction == self.LEFT:
            new_coordinate[1] = new_coordinate[1] - 1
        elif direction == self.RIGHT:
            new_coordinate[1] = new_coordinate[1] + 1
        
        return new_coordinate

    def can_snake_move_in_direction(self, direction):
        '''
        Helper function to check if it's possible to move in a certain direction
        Checks for:
        - If the snake is moving in the opposite direction as the way it's travelling
        (could be expanded if necessary)
        
        Parameters:
        -----------
        origin: (int, int)
            Coordinate to be moved

        Direction: int, options: [Snake.UP, Snake.DOWN, Snake.LEFT or Snake.RIGHT
            Direction to be moved

        Returns:
        -------
        coordinate: (int, int)
            Translated coordinate

        '''
        if self._is_facing_opposite_of_direction(direction):
            return False
        return True

    def is_head_outside_map(self):
        '''
        Returns a boolean indicating if the snake head is outside the map
        '''
        i_head, j_head = self.get_head()
        if 0 <= j_head < self.map_size[1]:
            if 0 <= i_head < self.map_size[0]:
                return False
        return True

    def get_snake_map(self, return_type="Binary"):
        '''
        Return an image including the positions of the snakes

        Parameter:
        ----------
        return_type: string
            if Binary, a binary image is returned
            if Colour, an image based on the snake's colour is returned
            if Numbered, an image with 1 as the head, 2, 3, 4 as the body
                is returned

        Returns:
        --------
        map_image, np.array(self.map_size)
            image of the position of this snake
        '''        
        if return_type == "Colour":
            map_image = np.zeros((self.map_size[0], self.map_size[1], 3))
        else:
            map_image = np.zeros((self.map_size[0], self.map_size[1]))

        if not self._is_alive or self.is_head_outside_map():
            # To check if the snake is dead or not
            return map_image

        for i, location in enumerate(self.locations):
            if return_type == "Colour":
                map_image[location[0], location[1], :] = self.colour
            elif return_type == "Binary":
                map_image[location[0], location[1]] = 1
            elif return_type == "Numbered":
                map_image[location[0], location[1]] = i+1

        # Color the head differently
        if return_type == "Colour":
            map_image[self.get_head()[0], self.get_head()[1], :] *= 0.5
        elif return_type == "Binary":
            map_image[self.get_head()[0], self.get_head()[1]] = 5

        return map_image

    def kill_snake(self):
        '''
        Set snake to be dead
        '''
        self._is_alive = False
        self.locations = []

    def is_alive(self):
        '''
        Get if the snake is alive
        '''
        return self._is_alive

    def get_size(self):
        '''
        Get the snake size
        '''
        return len(self.locations)

    def set_ate_food(self):
        '''
        Actions taken when the snake eaten food
        '''
        self.ate_food = True
        self.health = self.FULL_HEALTH

class Snakes:
    '''
    The Snakes class managers n number of snakes
    
    Parameters
    ----------
    map_size: (int, int)
    number_of_snakes: int
    
    snake_spawn_locations: [(int, int)] optional
        Parameter to force snakes to spawn in certain positions. Used for testing
    '''
    def __init__(self, map_size, number_of_snakes, snake_spawn_locations=[]):
        self.map_size = map_size
        self.number_of_snakes = number_of_snakes
        self.snakes = self._initialise_snakes(number_of_snakes, snake_spawn_locations)

    def _initialise_snakes(self, number_of_snakes, snake_spawn_locations):
        snakes = []

        if len(snake_spawn_locations) == 0:
            starting_positions = get_random_coordinates(self.map_size, number_of_snakes)
        else:
            error_message = "the number of coordinates in snake_spawn_locations must match the number of snakes"
            assert len(snake_spawn_locations) == self.number_of_snakes, error_message
            starting_positions = snake_spawn_locations

        for i in range(number_of_snakes):
            snakes.append(Snake(starting_position=starting_positions[i], map_size=self.map_size))

        return snakes

    @classmethod
    def make_from_dict(cls, map_size, snake_dicts):
        '''
        Class method to create the Snakes class from a dictionary of snakes

        Parameters
        ----------
        map_size: (int, int)
        snake_dicts: [{}]
            A list of snake_dict.
            dictionary are in the form of the battlesnake engine
        '''
        number_of_snakes = len(snake_dicts)
        cls = Snakes(map_size, number_of_snakes)
        cls.snakes = []
        
        for snake_dict in snake_dicts:
            locations = []

            for loc in snake_dict["body"]:
                locations.append((loc["y"], loc["x"]))
            
            health = snake_dict["health"]
            snake = Snake.make_from_list(locations, health, map_size)
            cls.snakes.append(snake)
        return cls

    def get_snake_51_map(self, excluded_snakes=[]):
        '''
        Function to generate a 51 map of the locations of any snake

        Parameters:
        ----------
        excluded_snakes: [Snake]
            Snakes to not be included in the binary map. 
            Used to check if there are collisions between snakes
        
        Returns:
        --------
        map_image: np.array(map_sizep[0], map_size[1], 1)
            If any snake is on coordinate i, j, map_image[i, j] will be 1
        '''
        sum_map = np.sum(self.get_snake_depth_51_map(excluded_snakes=excluded_snakes), 2)   
        return sum_map
            
    def get_snake_numbered_map(self, excluded_snakes=[]):
        '''
        Function to generate a numbered map of the locations of any snake
        1 will be the head, 2, 3 etc will be the body

        Parameters:
        ----------
        excluded_snakes: [Snake]
            Snakes to not be included in the binary map. 
            Used to check if there are collisions between snakes
        
        Returns:
        --------
        map_image: np.array(map_sizep[0], map_size[1], 1)
            If any snake is on coordinate i, j, map_image[i, j] will be 1
        '''
        return np.sum(self.get_snake_depth_numbered_map(
            excluded_snakes=excluded_snakes), 2)

    def get_snake_depth_numbered_map(self, excluded_snakes=[]):
        '''
        Function to generate a numbered map of the locations of any snake
        1 will be the head, 2, 3 etc will be the body

        Parameters:
        ----------
        excluded_snakes: [Snake]
            Snakes to not be included in the binary map. 
            Used to check if there are collisions between snakes

        Returns:
        --------
        map_image: np.array(map_sizep[0], map_size[1], number_of_snakes)
            The depth of the map_image corresponds to each snakes
            For each snake, 1 indicates the head and 2, 3, 4 etc indicates
             the body that the snake is present in that location and 0
            indicates that the snake is not present in that location
        '''
        map_image = np.zeros((self.map_size[0], self.map_size[1],
                              len(self.snakes)),
                             dtype=np.uint8)
        for i, snake in enumerate(self.snakes):
            if snake not in excluded_snakes:
                map_image[:, :, i] = snake.get_snake_map(return_type="Numbered")
        return map_image


    def get_snake_depth_51_map(self, excluded_snakes=[]):
        '''
        Function to generate a 51 map of the locations of the snakes

        Parameters:
        ----------
        excluded_snakes: [Snake]
            Snakes to not be included in the binary map. 
            Used to check if there are collisions between snakes

        Returns:
        --------
        map_image: np.array(map_sizep[0], map_size[1], number_of_snakes)
            The depth of the map_image corresponds to each snakes
            For each snake, 2 indicates the head and 1 indicates the body
             that the snake is present in that location and 0
            indicates that the snake is not present in that location
        '''
        map_image = np.zeros((self.map_size[0], self.map_size[1],
                              len(self.snakes)),
                             dtype=np.uint8)
        for i, snake in enumerate(self.snakes):
            if snake not in excluded_snakes:
                map_image[:, :, i] = snake.get_snake_map(return_type="Binary")

        return map_image

    def get_snake_colour_map(self):
        '''
        Function to generate a colour map of the locations of the snakes

        Returns:
        --------
        map_image: np.array(map_size[0], map_size[1], 3)
            The positions of the snakes are indicated by the colour of each snake
        '''
        map_image = np.zeros((self.map_size[0], self.map_size[1], 3))
        for snake in self.snakes:
            map_image += snake.get_snake_map(return_type="Colour")
        return map_image

    def get_snake_colours(self):
        '''
        The colours of each snake are provided
        '''
        snake_colours = []
        for snake in self.snakes:
            snake_colours.append(snake.colour)
        return snake_colours

    def move_snakes(self, action):
        '''
        Move the snakes based on action

        Parameters:
        ----------
        action: np.array(number_of_snakes)
            Array of integers containing an action for each number of snake. 
            The integers range from 0 to 3 corresponding to up, down, left, and right 
            respectively
        '''
        for i in range(len(action)):
            direction = action[i]
            self.snakes[i].move(direction)

    def get_snakes(self):
        '''
        Returns the list of snakes
        '''
        return self.snakes
