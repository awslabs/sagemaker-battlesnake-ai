from .snake import Snakes
from .food import Food
import pandas as pd
import string
import numpy as np

class Game_state_parser:
    '''
    Class to assist in parsing the game state provided in env.render(mode="ascii")
    
    Parameters:
    ----------
    filename: str
        Filepath to a textfile containing the rendered ascii file
    '''
    def __init__(self, filename):
        # Allows both - and | as delimiters of the text file
        self.input_data = pd.read_csv(filename, header=None, index_col=False, sep='-|\|', engine='python')
        self.map_size = self.get_map_size()
        self.number_of_snakes = self.get_number_of_snakes()
        
    def get_map_size(self):
        x = self.input_data.shape[1] - 2

        asterisks_count = 0
        for value in self.input_data[0]:
            if value is not None:
                if "*" in value:
                    asterisks_count += 1
        y = asterisks_count - 2
        return y, x
        
    def get_number_of_snakes(self):
        number_of_snakes = 0
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                grid_value = self.input_data.iloc[i+1, j+1]
                if grid_value is np.nan:
                    continue 
                if grid_value[0] in string.ascii_lowercase and grid_value[1] == '0':
                    number_of_snakes += 1
        return number_of_snakes

    def parse(self):
        snake_locations = {} # {Snake_identifier: (String: snake_identifier+index, i, j)}
        # Initialise snake_lcoations
        for i in range(self.number_of_snakes):
            snake_locations[string.ascii_lowercase[i]] = []
        
        food_locations = []
        snake_health = {}
        turn_count = None
        
        # Get the location of food and snakes
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                grid_value = self.input_data.iloc[i+1, j+1]
                if grid_value is np.nan:
                    continue
                if "@" in grid_value:
                    food_locations.append((i, j))
                if grid_value[0] in string.ascii_lowercase:
                    snake_identifier = grid_value[0]
                    import pdb; pdb.set_trace();                    
                    snake_locations[snake_identifier].append((grid_value[:2], i, j))

        # Get the health of snakes
        for x in range(self.number_of_snakes):
            health_line = self.input_data.iloc[-(x + 1), 0].replace(" ", "").split("=")
            snake_character = health_line[0]
            health = health_line[1]
            snake_health[snake_character] = int(health)

        # Get turn count
        idx = self.number_of_snakes + 1
        turn_count_line = self.input_data.iloc[-idx, 0].replace(" ", "").split("=")
        turn_count = int(turn_count_line[1])

        food = Food.make_from_list(self.map_size, food_locations)
        snakes = Snakes.make_from_lists(self.map_size, self.number_of_snakes, snake_locations,
                                       snake_health)
        return snakes, food, turn_count
