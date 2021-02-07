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

import json
import sys
import os
import random
import time
sys.path.append('./site-packages')

import numpy as np
import boto3
import botocore

from botocore.exceptions import ClientError
from convert_utils import ObservationToStateConverter

#################################
#              Init             #
#################################

converter = ObservationToStateConverter(style='one_versus_all', border_option="max")

config = botocore.config.Config(read_timeout=200)
runtime = boto3.client('runtime.sagemaker', config=config)

def proxyHandler(event, context):
    print("Request received")
    print(event)
    if (event["path"] == "/move"):
        if (event["httpMethod"] == "POST"):
            return move(event["body"])
        else:
            return {
                "statusCode": 403
            }
    elif (event["path"] == "/"):
        return ping()
    elif (event["path"] == "/start"):
        return start()
    elif (event["path"] == "/end"):
        return end()
    else:
        return {
            "statusCode": 404
        }

def ping():
    time.sleep(0.1)
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json"
        },
        "body": json.dumps({
            "apiversion": "1",
            "author": "my_user_name",
            "color": os.environ['SNAKE_COLOR'],
            "head": os.environ['SNAKE_HEAD'],
            "tail": os.environ['SNAKE_TAIL']
            })
        }

def start():
    time.sleep(0.1)
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json"
        }
    }

def move(body):
    data = json.loads(body)
    current_state, previous_state = converter.get_game_state(data)

    direction_index = remoteInferenceRLib(previous_state, current_state, data)

    directions = ['down', 'up', 'left', 'right']
    choice = directions[int(direction_index)]

    print("Move " + choice)

    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json"
        },
        "body": json.dumps({ "move": choice })
    }

def make_health_dict(data):
    health_dict = {0: data['you']['health']}
    for i, snake in enumerate(data["board"]["snakes"]):
        health_dict[i+1] = snake["health"]
    return health_dict

def remoteInferenceRLib(previous_state, current_state, data):
    map_width = data['board']['width']
    state = np.concatenate((previous_state, current_state), axis=2)
    state = np.expand_dims(state, 0)

    health_dict = make_health_dict(data)

    data = {"state": state.tolist(), "prev_action": -1,
            "prev_reward": -1, "seq_lens": -1,
            "action_mask": [1, 1, 1, 1],
            "all_health": health_dict, "json": data}

    payload = json.dumps(data)
    response = runtime.invoke_endpoint(EndpointName=os.environ['BATTLESNAKE_ENPOINT'],
                                       ContentType='application/json',
                                       Body=payload)
    direction_index = json.loads(response['Body'].read().decode())
    direction_index = direction_index["outputs"]["heuristisc_action"]
    return direction_index

def end():
    return {
        "statusCode": 200
    }
