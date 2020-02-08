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

converter = ObservationToStateConverter(style='one_versus_all', use_border=True)
net = mx.gluon.SymbolBlock.imports('model-symbol.json', 'data', 'model-0000.params')

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
    
    color = "#00FF00"
    time.sleep(0.1)
    return start_response(color)


@bottle.post('/move')
def move():
    data = bottle.request.json

    state = converter.get_game_state(data)
    
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
    if direction in possible:
        return move_response(direction)
    elif len(possible) > 0:
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
        port=os.getenv('PORT', '8080'),
        debug=os.getenv('DEBUG', True)
    )
