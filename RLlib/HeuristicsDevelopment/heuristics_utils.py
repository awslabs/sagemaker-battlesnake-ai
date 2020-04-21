import numpy as np
import tensorflow as tf
from rllib_src.utils import sort_states_for_snake_id

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

def build_state_for_snake(obs, snake_i, prev_state=None):
    '''
    Helper function that'll help the current state and previous state together
    If the previous state doesn't exist, append an empty state
    
    Parameters:
    ----------
    obs: np.array of size [map_size[0], map_size[1], 3]
    snake_i: int
        snake id
        
    Return: 
    -------
    output: np.array of size [map_size[0], map_size[1], 6]
    '''
    if prev_state is None:
        prev_state = np.zeros((obs.shape[0], obs.shape[1], 3))
        
    obs = np.array(obs, dtype=np.float32)
            
    obs = sort_states_for_snake_id(obs, snake_i+1)
    merged_map = np.concatenate((prev_state, obs), axis=-1)

    return merged_map, obs

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

def get_action(net, state):
    state = np.expand_dims(state, 0)
    predict = net(observations=tf.convert_to_tensor(state, dtype=tf.float32), 
               seq_lens=tf.constant([-1], dtype=tf.int32), 
               prev_action=tf.constant([-1], dtype=tf.int64),
               prev_reward=tf.constant([-1], dtype=tf.float32), 
               is_training=tf.constant(False, dtype=tf.bool))
    action = predict["behaviour_logits"].numpy()
    return action

def simulate(env, net, heuristics, number_of_snakes):    
    state, _, _, infos  = env.reset()

    rgb_arrays = [env.render(mode="rgb_array")]
    infos_array = [infos]
    actions_array = [[4, 4, 4, 4]]
    json_array = [env.get_json()]
        
    heuristics_log_array = [{k: "" for k in range(number_of_snakes)}]

    previous_state = {}
    for i in range(number_of_snakes):
        agent_id = "agent_{}".format(i)
        previous_state[agent_id] = None
        
    while True:
        infos["current_turn"] += 1

        heuristics_log = {}       
        actions = []
        for i in range(number_of_snakes):
            agent_id = "agent_{}".format(i)
            
            state_i, obs = build_state_for_snake(state, i, previous_state[agent_id])
            
            action = get_action(net, state_i)

            snake_list = make_snake_lists(env)
            map_size = env.map_size
            json = convert_state_into_json(map_size, state, snake_list, snake_id=i, 
                                           turn_count=infos["current_turn"]+1, 
                                           health=infos["snake_health"])
            # Add heuristics
            action, heuristics_log_string = heuristics.run(
                                                state, snake_id=i,
                                                turn_count=infos["current_turn"]+1,
                                                health=infos["snake_health"],
                                                json=json,
                                                action=action)
            heuristics_log[i] = heuristics_log_string
            
            actions.append(action)
            previous_state[agent_id] = obs
        
        next_state, reward, dones, infos = env.step(actions)
        
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
