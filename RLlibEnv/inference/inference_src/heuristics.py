import numpy as np

class Heuristics:
    def negative_heuristics(func):
        def negative_heuristics_func(self, *args, **kwargs):
            action = func(self, *args, **kwargs)
            assert np.sum(action) > 1, "A maximum of 2 False is allowed in the mask"
            return action
        return negative_heuristics_func

    def positive_heuristics(func):
        def positive_heuristics_func(self, *args, **kwargs):
            action = func(self, *args, **kwargs)
            assert np.sum(action) in [1, 4], "Only 1 true reward is allowed or all True are allowed"
            return action
        return positive_heuristics_func
    
    def _remove_borders_from_state(self, state, map_size):
        '''
        Helper function to remove the -1 borders from the state representation
        '''
        if -1 in state:
            y, x = map_size
            return state[int(y/2):-int(y/2), int(x/2):-int(x/2), :]
        else:
            return state
        
    def _convert_food_maxtrix_to_list(self, in_array):
        '''
        Helper function that converts a food matrix into a list of coordinates 
        containing food

        Parameters:
        ----------
        in_array: np.array of size [map_size[0], map_size[1], :]

        Return:
        -------
        food: [{"x": int, "y": int}]
        '''
        food = []
        y, x = np.where(in_array==1)
        for x_, y_ in zip(x, y):
            food.append({"x": x_, "y": y_})
        return food

    def _convert_state_into_json(self, map_size, state, snake_list, snake_id, turn_count, health):
        '''
        Helper function to build a JSON from the battlesnake gym.
        The JSON mimics the battlesnake engine
        '''
        FOOD_INDEX = 0

        state = self._remove_borders_from_state(state, map_size)
        food = self._convert_food_maxtrix_to_list(state[:, :, FOOD_INDEX])

        # Make snake_dict from snake_list
        snake_dict_list = []
        for i, snake in enumerate(snake_list):
            snake_dict = {}
            snake_dict["health"] = health[i]
            snake_dict["body"] = snake
            snake_dict["id"] = i
            snake_dict["name"] = "Snake {}".format(i)
            snake_dict_list.append(snake_dict)

        your_snake_dict_list = snake_dict_list[snake_id]
        other_snake_dict_list = snake_dict_list

        # Create board
        json = {}
        json["board"] = {"height": state.shape[0],
                        "width": state.shape[1],
                        "food": food, 
                        "snakes": other_snake_dict_list}
        json["you"] = your_snake_dict_list

        return json
    
    def _make_snake_lists(self, env):
        '''
        Helper function to create an ordered lists of snakes positions.

        Parameters:
        -----------
        env: BattlesnakeGym

        Returns:
        --------
        snake_locations: [[{"x": x, "y": y}]]
        This contains a list of the locations ([{"x": x, "y": y}]) for each snake. 
        '''
        snake_locations = []
        for snakes in env.snakes.snakes:
            snake_location = []
            for coord in snakes.locations[::-1]:
                snake_location.append({"x": coord[1], "y": coord[0]})
            snake_locations.append(snake_location)
        return snake_locations

    def run_with_env(self, state, snake_id, turn_count, health, action, env):
        '''
        Helper function to execute the run function with the BattlesnakeGym instead
        of json.
        '''
        snake_list = self._make_snake_lists(env)
        map_size = env.map_size
        json = self._convert_state_into_json(map_size, state, snake_list, snake_id, 
                                       turn_count, health)
        return self.run(state, snake_id, turn_count, health, json, action)
    
    def run_for_mask(self, functions_to_run):
        pass
        
    def run(self):
        raise NotImplementedError()