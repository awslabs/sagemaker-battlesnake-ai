from training.training_src.networks.utils import sort_states_for_snake_id
import numpy as np
import mxnet as mx
from collections import namedtuple

ctx = mx.gpu() if mx.context.num_gpus() > 0 else mx.cpu()

def remove_borders_from_state(state, map_size):
    '''
    Helper function to remove the -1 borders from the state representation
    '''
    if -1 in state:
        y, x = map_size
        return state[int(y/2):-int(y/2), int(x/2):-int(x/2), :]
    else:
        return state
    
def convert_food_maxtrix_to_list(in_array):
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

def make_snake_lists(env):
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

def convert_state_into_json(map_size, state, snake_list, snake_id, turn_count, health):
    '''
    Helper function to build a JSON from the battlesnake gym.
    The JSON mimics the battlesnake engine
    '''
    FOOD_INDEX = 0
    
    state = remove_borders_from_state(state, map_size)
    food = convert_food_maxtrix_to_list(state[:, :, FOOD_INDEX])
    
    # Make snake_dict from snake_list
    snake_dict_list = []
    for i, snake in enumerate(snake_list):
        snake_dict = {}
        snake_dict["health"] = health[i]
        snake_dict["body"] = snake
        snake_dict["id"] = snake_id
        snake_dict["name"] = "Snake {}".format(snake_id)
        snake_dict_list.append(snake_dict)
        
    your_snake_dict_list = snake_dict_list[snake_id]
    del snake_dict_list[snake_id]
    other_snake_dict_list = snake_dict_list
    
    # Create board
    json = {}
    json["board"] = {"height": state.shape[0],
                    "width": state.shape[1],
                    "food": food, 
                    "snakes": other_snake_dict_list}
    json["you"] = your_snake_dict_list
        
    return json

def get_action(net, state, snake_id, turn_count, health, memory):
    '''
    Processes the input data to be fed into the defined neural network.

    Parameters:
    -----------
    `net`: gluon.sequential
    model of the neural net
    
    `state`: np.array of size (3, map_size[0]+2, map_size[1]+2)
    Provides the current observation of the gym
    
    `snake_id`: int
    Indicates the id where id \in [0...number_of_snakes]
    
    `turn_count`: int
    Indicates the number of elapsed turns
    
    `health`: dict
    Indicates the health of all snakes in the form of {snake_id: health}
    
    `memory`: (state, turn_count, health)
    Indicates the state, turn_count, and health of the previous turn.
    
    Returns:
    -----------
    `action`: np.array 
    The expected q value of the action.
    i.e., the larger the value the better the action is.
    To get the next best action, perform np.argmax(action)
    '''
    
    sequence_length = 2
    state_i = sort_states_for_snake_id(state, snake_id+1, one_versus_all=True)
    previous_state_i = sort_states_for_snake_id(memory.state, snake_id+1, one_versus_all=True)
    state_sequence = mx.nd.array(np.stack([previous_state_i, state_i]), ctx=ctx).transpose((0, 3, 1, 2)).expand_dims(0)
    
    snake_id_sequence = mx.nd.array([snake_id]*sequence_length, ctx=ctx).expand_dims(0)
    turn_count_sequence = mx.nd.array([memory.turn_count, turn_count], ctx=ctx).expand_dims(0)
    snake_health_sequence = mx.nd.array([memory.health[snake_id], health[snake_id]], ctx=ctx).expand_dims(0)
        
    action = net(state_sequence, snake_id_sequence, turn_count_sequence, snake_health_sequence)
    '''
    Net takes the following arguments:
    `net(state, snake_id, turn_count, snake_health)`

    `state`: *nd.array* of size (batch_size, sequence_length=2, c=3, map_size[0]+2, mapsize[1]+2)
    - Provides the observation space of the gym
    - `batch_size` should be set to 1
    - `sequence_length` provides the number of timesteps back the model considers. Give `t` is the current time step, `c=0`   refers to `t-1` and `c=1` is t.
    - Each `c` slide refers to the *food*, *current snake*, and *other snakes* respectively.
    - `map_size` is based on the size of the BattlesnakeGym +2 for the -1 border.

    `snake_id`: *nd.array* of size (batch_size, sequence_length=2)
    - Provides the id of the snake, which is a i \in [0...number_of_snakes-1]

    `turn_count`: *nd.array* of size (batch_size, sequence_length=2)
    - Provides the number of turns that has elapsed (obtained from `info["current_turn"]` in the gym)

    `snake_health`: *nd.array* of size (batch_size, sequence_length=2)
    - Provides the health of the snake (obtained from `info["snake_health"]` in the gym)
    '''
    return action.asnumpy()[0]

def is_snake_alive(env, snake_id):
    return env.snakes.get_snakes()[snake_id].is_alive()

def simulate(env, net, heuristics, number_of_snakes):
    '''
    Helper functions to simulate the snakes moving with BattlesnakeGym.
    Pseudo code is:
    Until only 1 snake is alive:
        1. Get actions from the neural network
        2. Run heuristisc
        3. Feed into the battlesnake gym
    '''
    Memory = namedtuple("Memory", "state turn_count health")
    
    state, _, _, infos = env.reset()
    
    rgb_arrays = [env.render(mode="rgb_array")]
    infos_array = [infos]
    actions_array = [[4, 4, 4, 4]]
    json_array = [env.get_json()]
        
    heuristics_log_array = [{k: "" for k in range(number_of_snakes)}]

    memory = Memory(state=np.zeros(state.shape), turn_count=infos["current_turn"], health=infos["snake_health"])
    while True:
        infos["current_turn"] += 1

        heuristics_log = {}
        actions = []
        for i in range(number_of_snakes):
            action = get_action(net, state, snake_id=i,
                                turn_count=infos["current_turn"]+1,
                                health=infos["snake_health"],
                                memory=memory)

            snake_list = make_snake_lists(env)
            map_size = env.map_size
            json = convert_state_into_json(map_size, state, snake_list, snake_id=i, 
                                           turn_count=infos["current_turn"]+1, 
                                           health=infos["snake_health"])
            # Add heuristics
            if is_snake_alive(env, i):
                action, heuristics_log_string = heuristics.run(
                                                state, snake_id=i,
                                                turn_count=infos["current_turn"]+1,
                                                health=infos["snake_health"],
                                                json=json,
                                                action=action)           
            else:
                action = np.argmax(action[0])
                heuristics_log_string = "Dead"
            heuristics_log[i] = heuristics_log_string
            
            actions.append(action)
        memory = Memory(state=state, turn_count=infos["current_turn"], 
                        health=infos["snake_health"])
        
        next_state, reward, dones, infos = env.step(np.array(actions))
        
        rgb_array = env.render(mode="rgb_array")
        rgb_arrays.append(rgb_array.copy())
        infos_array.append(infos)
        actions_array.append(actions)
        heuristics_log_array.append(heuristics_log)
        json_array.append(env.get_json())
        
        # Check if only 1 snake remains
        number_of_snakes_alive = sum(list(dones.values()))
        if number_of_snakes - number_of_snakes_alive <= 1:
            done = True
        else:
            done = False

        state = next_state
        if done:
            print("Completed")
            break  

    return infos_array, rgb_arrays, actions_array, heuristics_log_array, json_array