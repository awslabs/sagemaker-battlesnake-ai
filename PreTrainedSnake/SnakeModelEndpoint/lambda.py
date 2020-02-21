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

from convert_utils import ObservationToStateConverter

#################################
#              Init             #
#################################

converter = ObservationToStateConverter(style='one_versus_all', use_border=True)
ctx = mx.gpu() if mx.context.num_gpus() > 0 else mx.cpu()
nets = {str(k):mx.gluon.SymbolBlock.imports('local-run-{}x{}-symbol.json'.format(k,k), ['data0', 'data1', 'data2', 'data3'], 'local-run-{}x{}-0000.params'.format(k,k), ctx=ctx) for k in [7,11,15,19]}
[net.hybridize(static_alloc=True, static_shape=True) for net in nets.values()]

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
    
    # Get the list of possible directions
    i,j = np.unravel_index(np.argmax(current_state[:,:,1], axis=None), current_state[:,:,1].shape)
    snakes = current_state[:,:,1:].sum(axis=2)
    possible = []
    if snakes[i+1,j] == 0:
        possible.append('down')
    if snakes[i-1,j] == 0:
        possible.append('up')
    if snakes[i,j+1] == 0:
        possible.append('right')
    if snakes[i,j-1] == 0:
        possible.append('left')

    # Sending the current_states for inference
    current_state_nd = mx.nd.array(current_state, ctx=ctx)
    previous_state_nd = mx.nd.array(previous_state, ctx=ctx)
    current_state_nd = current_state_nd.expand_dims(axis=0).transpose((0, 3, 1, 2)).expand_dims(axis=1)
    previous_state_nd = previous_state_nd.expand_dims(axis=0).transpose((0, 3, 1, 2)).expand_dims(axis=1)
    
    current_state_nd = mx.nd.concatenate([previous_state_nd, current_state_nd], axis=1)
    turn_sequence = mx.nd.array([data['turn']]*2, ctx=ctx).reshape((1,-1))
    health_sequence = mx.nd.array([data['you']['health']]*2, ctx=ctx).reshape((1,-1))

    net = nets[str(data['board']['width'])]
    # Getting the result from the model
    output = sum([net(current_state_nd, mx.nd.array([i]*2, ctx=ctx).reshape((1,-1)), turn_sequence, health_sequence).softmax() for i in range(4)])
    
    # Getting the highest predicted index
    direction_index = output.argmax(axis=1)[0].asscalar()
    
    directions = ['up', 'down', 'left', 'right']
    direction = directions[int(direction_index)]
    choice = direction

    if direction in possible:
        # Don't starve if possible
        if data['you']['health'] < 30 and len(food_locations) > 0 and direction not in food_locations:
            print("eating food instead of move")
            choice = random.choice(food_locations)
    elif len(possible) > 0:
        # Don't starve if possible        
        if data['you']['health'] < 30 and len(food_locations) > 0 and direction not in food_locations:
            print("eating food instead of dying")
            choice = random.choice(food_locations)
        # Don't kill yourself
        else:
            print("Move "+direction+" is not possible")
            choice = random.choice(possible)

    print("Move " + choice)

    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json"
        },
        "body": json.dumps({ "move": choice })
    }


def end():
    return {
        "statusCode": 200
    }
