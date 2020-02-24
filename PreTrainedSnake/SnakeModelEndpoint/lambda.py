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

from convert_utils import ObservationToStateConverter

#################################
#              Init             #
#################################

converter = ObservationToStateConverter(style='one_versus_all', use_border=True)
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
    
    state = np.expand_dims(np.stack([previous_state, current_state]), 0).transpose(0, 1, 4, 2, 3)
    snake_id = np.array([[1]]*2).transpose(1, 0)
    turn = np.array([[data['turn']-1], [data['turn']]]).transpose(1, 0)
    health = np.array([[data['you']['health']-1], [data['you']['health']]]).transpose(1, 0)
    data = {"state": state, "snake_id": snake_id, "turn_count": turn, "health": health}
    payload = json.dumps(data)
    response = runtime.invoke_endpoint(EndpointName="battlesnake-endpoint",
                                       ContentType='application/json',
                                       Body=payload)
    direction_index = json.loads(response['Body'].read().decode())

    directions = ['up', 'down', 'left', 'right']
    direction = directions[int(direction_index)]
    choice = direction

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
