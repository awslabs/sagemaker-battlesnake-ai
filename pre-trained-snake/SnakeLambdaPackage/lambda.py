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
net = mx.gluon.SymbolBlock.imports('model-symbol.json', 'data', 'model-0000.params')

def ping(event, context):
    return {
        "statusCode": 200
    }

def start(event, context):
    print("Start")
    color = "#00FF00"
    time.sleep(0.1)
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json"
        },
        "body": json.dumps({ "color": color })
    }

def move(event, context):
    print("Move")
    print(event)

    state = converter.get_game_state(json.loads(event['body']))
    
    # Get the list of possible directions
    i,j = np.unravel_index(np.argmax(state[:,:,1], axis=None), state[:,:,1].shape)
    snakes = state[:,:,1:].sum(axis=2)
    possible = []
    if snakes[i+1,j] == 0:
        possible.append('down')
    if snakes[i-1,j] == 0:
        possible.append('up')
    if snakes[i,j+1] == 0:
        possible.append('right')
    if snakes[i,j-1] == 0:
        possible.append('left')

    # Sending the states for inference
    state_nd = mx.nd.array(state)
    state_nd = state_nd.expand_dims(axis=0).transpose((0, 3, 1, 2))
    
    # Getting the result from the model
    output = net(state_nd)
    
    # Getting the highest predicted index
    direction_index = output.argmax(axis=1)[0].asscalar()
    
    directions = ['up', 'down', 'left', 'right']
    direction = directions[int(direction_index)]
    choice = direction
    if direction not in possible:
      print("Move "+direction+" is not possible")
      if len(possible) > 0:
        choice = random.choice(possible)

    print("Move " + choice)

    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json"
        },
        "body": json.dumps({ "move": choice })
    }


def end(event, context):
    return {
        "statusCode": 200
    }
