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
import os
import random
import time
import mxnet as mx
import numpy as np
import boto3
import botocore

from battlesnake_heuristics import MyBattlesnakeHeuristics
from convert_utils import ObservationToStateConverter

#################################
#              Init             #
#################################

converter = ObservationToStateConverter(style='one_versus_all', use_border=True)
config = botocore.config.Config(read_timeout=200)

useSageMakerEndpoint = os.environ['USE_SAGEMAKER_ENDPOINT']

if (useSageMakerEndpoint == "true"):
    runtime = boto3.client('runtime.sagemaker', config=config)
else:
    ctx = mx.gpu() if mx.context.num_gpus() > 0 else mx.cpu()
    nets = {}
    for k in [7, 11, 15, 19]:
        symbol = 'Model-{}x{}/local-symbol.json'.format(k,k)
        params = 'Model-{}x{}/local-0000.params'.format(k, k)
        nets[str(k)] = mx.gluon.SymbolBlock.imports(symbol, ['data0', 'data1', 'data2', 'data3'], params)
    [net.hybridize(static_alloc=True, static_shape=True) for net in nets.values()]
    heuristics = MyBattlesnakeHeuristics()


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
    elif (event["path"] == "/start"):
        return start()
    elif (event["path"] == "/ping"):
        return ping()
    elif (event["path"] == "/end"):
        return end()
    else:
        return {
            "statusCode": 404
        }

def ping():
    return {
        "statusCode": 200
    }

def start():
    time.sleep(0.1)
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json"
        },
        "body": json.dumps({
            "color": os.environ['SNAKE_COLOR'],
            "headType": os.environ['SNAKE_HEAD'],
            "tailType": os.environ['SNAKE_TAIL']
        })
    }


def move(body):
    data = json.loads(body)
    current_state, previous_state = converter.get_game_state(data)

    if (useSageMakerEndpoint == "true"):
        direction_index = remoteInference(previous_state, current_state, data)
    else:
        direction_index = localInference(previous_state, current_state, data)

    directions = ['up', 'down', 'left', 'right']
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

def localInference(previous_state, current_state, data):

    # Sending the states for inference
    current_state_nd = mx.nd.array(current_state, ctx=ctx)
    previous_state_nd = mx.nd.array(previous_state, ctx=ctx)
    current_state_nd = current_state_nd.expand_dims(axis=0).transpose((0, 3, 1, 2)).expand_dims(axis=1)
    previous_state_nd = previous_state_nd.expand_dims(axis=0).transpose((0, 3, 1, 2)).expand_dims(axis=1)
    
    state_nd = mx.nd.concatenate([previous_state_nd, current_state_nd], axis=1)
    turn_sequence = mx.nd.array([data['turn']-1, data['turn']], ctx=ctx).reshape((1,-1))
    health_sequence = mx.nd.array([data['you']['health']-1, data['you']['health']], ctx=ctx).reshape((1,-1))
    id_sequence = mx.nd.array([1]*2, ctx=ctx).reshape((1,-1))

    net = nets[str(data['board']['width'])]

    # Getting estimation of all direction from the model
    action = net(state_nd, id_sequence, turn_sequence, health_sequence)
    
    health_dict = make_health_dict(data)
    # Invoke heuristics to take final decision
    direction_index = heuristics.run(current_state, 1, data['turn'], health_dict, output)

    return direction_index
 

def remoteInference(previous_state, current_state, data):
    
    state = np.expand_dims(np.stack([previous_state, current_state]), 0).transpose(0, 1, 4, 2, 3)
    snake_id = np.array([[1]]*2).transpose(1, 0)
    turn = np.array([[data['turn']-1], [data['turn']]]).transpose(1, 0)
    health = np.array([[data['you']['health']+1], [data['you']['health']]]).transpose(1, 0)
    map_width = data['board']['width']

    health_dict = make_health_dict(data)
    print("health dict {}".format(health_dict))

    data = {"state": state.tolist(), "snake_id": snake_id.tolist(), 
        "turn_count": turn.tolist(), "health": health.tolist(),
        "all_health": health_dict, "map_width": map_width}
    payload = json.dumps(data)
    response = runtime.invoke_endpoint(EndpointName="battlesnake-endpoint",
                                       ContentType='application/json',
                                       Body=payload)
    direction_index = json.loads(response['Body'].read().decode())

    return direction_index


def end():
    return {
        "statusCode": 200
    }
