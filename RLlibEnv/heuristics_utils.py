import numpy as np
import tensorflow as tf
from training.training_src.utils import sort_states_for_snake_id
    

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
    
    if obs.shape[2] == 2:
        empty_state_with_borders = obs[:, :, 0]
        empty_state_with_borders[empty_state_with_borders==1] = 0
        empty_state_with_borders = np.expand_dims(empty_state_with_borders, 2)
        obs_i = np.concatenate((obs, empty_state_with_borders), axis=-1)
    else:    
        obs_i = sort_states_for_snake_id(obs, snake_i+1)
    
    merged_map = np.concatenate((prev_state, obs_i), axis=-1)

    return merged_map, obs_i

def is_snake_alive(env, snake_id):
    return env.snakes.get_snakes()[snake_id].is_alive()

def get_action(net, state, prev_action, prev_reward):
    state = np.expand_dims(state, 0)
    
    if prev_action == None:
        prev_action = -1
    if prev_reward == None:
        prev_reward = -1

    obs = {"obs": tf.convert_to_tensor(state, dtype=tf.float32),
            "action_mask": tf.convert_to_tensor([1, 1, 1, 1], dtype=tf.float32)}
    input_obs = tf.reshape(obs["obs"], shape=(1*21*21*6, )) 
    input_obs = tf.concat([obs["action_mask"], input_obs], axis=0)
    input_obs = tf.expand_dims(input_obs, 0)
            
    predict = net(observations=input_obs,
               seq_lens=tf.constant([60], dtype=tf.int32), 
               prev_action=tf.constant([prev_action], dtype=tf.int64),
               prev_reward=tf.constant([prev_reward], dtype=tf.float32), 
               is_training=tf.constant(False, dtype=tf.bool))
    action = predict["behaviour_logits"].numpy()
    return action

def simulate(env, net, heuristics, number_of_snakes, use_random_snake):    
    state, _, _, infos  = env.reset()

    rgb_arrays = [env.render(mode="rgb_array")]
    infos_array = [infos]
    actions_array = [[4 for _ in range(number_of_snakes)]]
    json_array = [env.get_json()]
        
    heuristics_log_array = [{k: "" for k in range(number_of_snakes)}]

    previous_move = {}
    for i in range(number_of_snakes):
        agent_id = "agent_{}".format(i)
        previous_move[agent_id] = {"state": None,
                                   "reward": None,
                                   "action": None}
        
    while True:
        infos["current_turn"] += 1

        heuristics_log = {}       
        actions = []
        for i in range(number_of_snakes):
            agent_id = "agent_{}".format(i)
            
            state_i, obs = build_state_for_snake(state, i, previous_move[agent_id]["state"])
            
            if use_random_snake:
                action = np.random.uniform(size=(1, 4))
            else:
                action = get_action(net, state_i, previous_move[agent_id]["action"],
                                    previous_move[agent_id]["reward"])
            
            if is_snake_alive(env, i):
                action, heuristics_log_string = heuristics.run_with_env(
                                                    state_i, snake_id=i,
                                                    turn_count=infos["current_turn"]+1,
                                                    health=infos["snake_health"],
                                                    env=env,
                                                    action=action)
            else:
                action = np.argmax(action[0])
                heuristics_log_string = "Dead"
            
            heuristics_log[i] = heuristics_log_string
            
            actions.append(action)
        
        next_state, rewards, dones, infos = env.step(actions)
        
        for i in range(number_of_snakes):
            agent_id = "agent_{}".format(i)
            action = actions[i]
            reward = rewards[i]
            previous_move[agent_id] = {"state": obs,
                                       "reward": reward,
                                       "action": action}

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
        
        if number_of_snakes == 1:
            snakes_to_win = 0
        else:
            snakes_to_win = 1
            
        if len(np.where(np.sum(next_state, axis=2)==5)[0]) == snakes_to_win:
            done = True
        else:
            done = False
            
        state = next_state
        if done:
            print("Completed")
            break  

    return infos_array, rgb_arrays, actions_array, heuristics_log_array, json_array
