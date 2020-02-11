import argparse
import json
import os
import random
import bottle
import time

from api import ping_response, start_response, move_response, end_response
import mxnet as mx
import numpy as np

from convert_utils import ObservationToStateConverter

#################################
#              Init             #
#################################

parser = argparse.ArgumentParser(
        description='Start the model for battlesnake API')

parser.add_argument('--port',type=int, default=8080, metavar="S", help='port for the API')
args = parser.parse_args()
port = args.port

converter = ObservationToStateConverter(style='one_versus_all', use_border=True)
ctx = mx.gpu() if mx.context.num_gpus() > 0 else mx.cpu()
nets = {str(k):mx.gluon.SymbolBlock.imports('local-run-{}x{}-symbol.json'.format(k,k), ['data0', 'data1', 'data2', 'data3'], 'local-run-{}x{}-0000.params'.format(k,k), ctx=ctx) for k in [7,11,15,19]}
[net.hybridize(static_alloc=True, static_shape=True) for net in nets.values()]
@bottle.route('/')
def index():
    return '''
    Battlesnake documentation can be found at
       <a href="https://docs.battlesnake.io">https://docs.battlesnake.io</a>.
    '''

@bottle.route('/static/<path:path>')
def static(path):
    """
    Given a path, return the static file located relative
    to the static folder.
    This can be used to return the snake head URL in an API response.
    """
    return bottle.static_file(path, root='static/')

@bottle.post('/ping')
def ping():
    """
    A keep-alive endpoint used to prevent cloud application platforms,
    such as Heroku, from sleeping the application instance.
    """
    return ping_response()

@bottle.post('/start')
def start():
    data = bottle.request.json

    """
    TODO: If you intend to have a stateful snake AI,
            initialize your snake state here using the
            request's data if necessary.
    """
    state = converter.get_game_state(data)
    
    color = "#000088"
    return start_response(color)


@bottle.post('/move')
def move():
    data = bottle.request.json

    current_state, previous_state = converter.get_game_state(data)
    
    # Get the list of possible directions
    i,j = np.unravel_index(np.argmax(current_state[:,:,1], axis=None), current_state[:,:,1].shape)
    snakes = current_state[:,:,1:].sum(axis=2)
    food = current_state[:,:,0]
    possible = []
    food_locations = []
    if snakes[i+1,j] == 0:
        possible.append('down')
    if snakes[i-1,j] == 0:
        possible.append('up')
    if snakes[i,j+1] == 0:
        possible.append('right')
    if snakes[i,j-1] == 0:
        possible.append('left')
    
    # Food locations
    if food[i+1, j] == 1:
        food_locations.append('down')
    if food[i-1,j] == 1:
        food_locations.append('up')
    if food[i,j+1] == 1:
        food_locations.append('right')
    if food[i,j-1] == 1:
        food_locations.append('left')
        

    # Sending the states for inference
    current_state_nd = mx.nd.array(current_state, ctx=ctx)
    previous_state_nd = mx.nd.array(previous_state, ctx=ctx)
    current_state_nd = current_state_nd.expand_dims(axis=0).transpose((0, 3, 1, 2)).expand_dims(axis=1)
    previous_state_nd = previous_state_nd.expand_dims(axis=0).transpose((0, 3, 1, 2)).expand_dims(axis=1)
    
    state_nd = mx.nd.concatenate([previous_state_nd, current_state_nd], axis=1)
    turn_sequence = mx.nd.array([data['turn']]*2, ctx=ctx).reshape((1,-1))
    health_sequence = mx.nd.array([data['you']['health']]*2, ctx=ctx).reshape((1,-1))

    net = nets[str(data['board']['width'])]
    # Getting the result from the model
    output = sum([net(state_nd, mx.nd.array([i]*2, ctx=ctx).reshape((1,-1)), turn_sequence, health_sequence).softmax() for i in range(4)])
    
    # Getting the highest predicted index
    direction_index = output.argmax(axis=1)[0].asscalar()
    
    directions = ['up', 'down', 'left', 'right']
    direction = directions[int(direction_index)]
    if direction in possible:
        # Don't starve if possible
        if data['you']['health'] < 30 and len(food_locations) > 0 and direction not in food_locations:
            print("eating food instead of move")
            choice = random.choice(food_locations)
            return move_response(choice)
        return move_response(direction)
    elif len(possible) > 0:
        # Don't starve if possible        
        if data['you']['health'] < 30 and len(food_locations) > 0 and direction not in food_locations:
            print("eating food instead of dying")
            choice = random.choice(food_locations)
        # Don't kill yourself
        else:
            choice = random.choice(possible)
        return move_response(choice)
    else:
        return move_response(direction)


@bottle.post('/end')
def end():
    data = bottle.request.json

    """
    TODO: If your snake AI was stateful,
        clean up any stateful objects here.
    """

    return end_response()

# Expose WSGI app (so gunicorn can find it)
application = bottle.default_app()

if __name__ == '__main__':
    bottle.run(
        application,
        host=os.getenv('IP', '0.0.0.0'),
        port=os.getenv('PORT', port),
        debug=os.getenv('DEBUG', True)
    )
