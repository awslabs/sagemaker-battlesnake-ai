import numpy as np

def sort_states_for_snake_id(state, snake_id, one_versus_all):
    '''
    Given states of shape (m, n, s+1) where m and n is the dimension of the map
    and s+1 is the number of snakes + 1 for food, sort the state depth so that the "self" snake
    is at index 1
    
    Params:
    -------
    state: np.array(m, n, s)
    snake_id: int
        ID of the state to be placed at index 1
    one_versus_all: Bool
        Should convert all other states into 1 layer
    '''    
    food_state = state[:, :, 0]
    self_state = state[:, :, snake_id]

    other_states = []
    for i in range(1, state.shape[2]):
        if i == snake_id:
            continue
        other_states.append(state[:, :, i])
        
    other_states = np.stack(other_states, axis=2)

    if one_versus_all:
        output_states = np.zeros(shape=(state.shape[0],
                                        state.shape[1],
                                        3))
        output_states[:, :, 0] = food_state
        output_states[:, :, 1] = self_state
        if other_states[0, 0, 0] == -1: # if states are bordered
            output_states[:, :, 2] = -1

            # Find all values excluding the border, assuming borders are -1
            other_states = np.sum(other_states, axis=2)
            other_states[other_states<0] = -1

            output_states[:, :, 2] = other_states
        else:
            output_states[:, :, 2] = np.sum(other_states, axis=2)

    else:
        output_states = np.zeros(shape=state.shape)
        output_states[:, :, 0] = food_state
        output_states[:, :, 1] = self_state
        output_states[:, :, 2:] = other_states

    return output_states
