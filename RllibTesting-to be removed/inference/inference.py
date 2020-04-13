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
        # pass through json (assumes it's correctly formed)
        d = data.read().decode('utf-8')
        return d if len(d) else ''

    if context.request_content_type == 'text/csv':
        # very simple csv handler
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
        
    heuristics_state = np.array(input_dict['inputs']['observations'])
    heuristics_id = 0
    heuristics_turn = input_dict['json']["turn"]
    heuristics_json = input_dict['json']
    heuristics_health = input_dict["all_health"]

    converted_heuristic_health = {}
    for k in heuristics_health:
        converted_heuristic_health[int(k)] = heuristics_health[k]

    converted_action, _ = heuristics.run(heuristics_state, 
                                      int(heuristics_id),
                                      int(heuristics_turn)+1,
                                      converted_heuristic_health, 
                                      json=heuristics_json,
                                      action=action_probs)

    prediction_dict["outputs"]["heuristisc_action"] = converted_action
    
    prediction_output = json.dumps(prediction_dict)
    
    return prediction_output, response_content_type
