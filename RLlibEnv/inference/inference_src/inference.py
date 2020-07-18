import json
import requests
import numpy as np

from battlesnake_heuristics import MyBattlesnakeHeuristics

heuristics = MyBattlesnakeHeuristics()

def handler(data, context):
    """Handle request.
    Args:
        data (obj): the request data
        context (Context): an object containing request and configuration details
    Returns:
        (bytes, string): data to return to client, (optional) response content type
    """
    processed_input = _process_input(data, context)
    response = requests.post(context.rest_uri, data=processed_input)
    return _process_output(response, processed_input, context)

def _process_input(data, context):
    if context.request_content_type == 'application/json':
        data = json.loads(data.read().decode('utf-8'))
        
        # Convert json for SageMaker RL format        
        observations = np.array(data["state"])
        action_mask = data["action_mask"]
        prev_action = data["prev_action"]
        prev_reward = data["prev_reward"]
        seq_lens = data["seq_lens"]
        health_dict = data["all_health"]
        json_ = data["json"]
        
        obs = np.concatenate([action_mask, observations.reshape((-1))], axis=0)
        obs = np.expand_dims(obs, 0)
        
        d = {"inputs": { 'observations': obs.tolist(),
                         'prev_action': prev_action,
                         'is_training': False,
                         'prev_reward': prev_reward,
                         'seq_lens': seq_lens
                     },
                 "all_health": health_dict,
                 "json": json_,
                  "state": observations.tolist()
            }

        d = json.dumps(d)
        return d if len(d) else ''

    if context.request_content_type == 'text/csv':
        return json.dumps({
            'instances': [float(x) for x in data.read().decode('utf-8').split(',')]
        })

    raise ValueError('{{"error": "unsupported content type {}"}}'.format(
        context.request_content_type or "unknown"))

def _process_output(data, input_data, context):
    if data.status_code != 200:
        raise ValueError(data.content.decode('utf-8'))
        
    response_content_type = context.accept_header
    prediction = data.content
    
    prediction_dict = json.loads(prediction)
    action_probs = prediction_dict['outputs']["behaviour_logits"]
    
    input_dict = json.loads(input_data)
    
    heuristics_state = np.array(input_dict['state'])
    heuristics_state = heuristics_state[0, :, :, 3:]
    heuristics_id = 0
    heuristics_turn = input_dict['json']["turn"]
    heuristics_json = input_dict['json']
    heuristics_health = input_dict["all_health"]

    converted_heuristic_health = {}
    for k in heuristics_health:
        converted_heuristic_health[int(k)] = heuristics_health[k]

    converted_action, log_string = heuristics.run(heuristics_state, 
                                      int(heuristics_id),
                                      int(heuristics_turn)+1,
                                      converted_heuristic_health, 
                                      json=heuristics_json,
                                      action=action_probs)
    print("Action {} Heuristics log {} {}".format(action_probs, log_string, converted_action))
    prediction_dict["outputs"]["heuristisc_action"] = converted_action
    
    prediction_output = json.dumps(prediction_dict)
    
    return prediction_output, response_content_type
