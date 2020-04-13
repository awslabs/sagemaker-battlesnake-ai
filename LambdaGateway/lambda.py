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
import numpy as np
import boto3
import botocore

from botocore.exceptions import ClientError
from convert_utils import ObservationToStateConverter

#################################
#              Init             #
#################################

if os.environ['SELECTED_RL_METHOD'] == "MXNet":
    converter = ObservationToStateConverter(style='one_versus_all', border_option="1")
else:
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
    elif (event["path"] == "/start"):
        return start()
    elif (event["path"] == "/ping"):
        return ping()
    elif (event["path"] == "/end"):
        return end()
    elif (event["path"] == "/status"):
        return status()
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

def status():
    time.sleep(0.1)

    # Check if inference endpoint is available or not
    status = "unknown"
    endpoint_status = "unknown"
    endpoint_name = 'battlesnake-endpoint'
    client = boto3.client('sagemaker')
    try:
        response = client.describe_endpoint(EndpointName=endpoint_name)
        endpoint_status = response['EndpointStatus']
    except ClientError as e:
        error_code = e.response['Error']['Code']
        print(e.response['Error'])
        endpoint_status = 'Endpoint does not exist'

    html = "<html><head><title>Snake status</title></head><body>"

    if( endpoint_status == 'InService'):
        status = "ready"
        html += "<b>snake status : </b>" + status
    else:
        if( endpoint_status == 'Endpoint does not exist'):
            status = "deployment failed"
            html += "<b>snake status : </b>" + status + "<br><br>"
            html += "<b>Amazon SageMaker endpoint status : </b>" + endpoint_status + "<br><br>"
            html += "<i>The Amazon SageMaker endpoint creation have failed. This is probably because your account cannot launch ml.m5.xlarge instance yet.</i><br><br>"
            html += "<b>What you should fo now:</b><br><br>"
            html += "1. Go in CloudFormation service in the AWS console and delete the stack<br>"
            html += "2. Recreate the stack following one of the two options below:<br>"
            html += "<ul><li>Click on the 'deploy' link from Github and before clicking 'create stack' change the selector SagemakerInstanceType to ml.t2.medium. This instance type is allowed by default in every account. Note that it is not free.</li>"
            html += "<li>Open a support ticket and ask to be allowed to launch ml.m5.xlarge instances. Once approved, recreate the stack with default settings</li></ul>"
        else:
            status = "not ready"
            html += "<b>snake status : </b>" + status + "<br><br>"
            html += "<b>Amazon SageMaker endpoint status : </b>" + endpoint_status + "<br><br>"
            html += "<i>You can visit the Amazon SageMaker service page in the AWS Console to see detailed information.</i>"

    html += "</body></html>"

    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "text/html"
        },
        "body": html
    }

def move(body):
    data = json.loads(body)
    current_state, previous_state = converter.get_game_state(data)

    if os.environ['SELECTED_RL_METHOD'] == "MXNet":
        direction_index = remoteInferenceMXNet(previous_state, current_state, data)
    else:
        direction_index = remoteInferenceRLib(previous_state, current_state, data)
        
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

def remoteInferenceMXNet(previous_state, current_state, data):
    state = np.expand_dims(np.stack([previous_state, current_state]), 0).transpose(0, 1, 4, 2, 3)
    snake_id = np.array([[1]]*2).transpose(1, 0)
    turn = np.array([[data['turn']-1], [data['turn']]]).transpose(1, 0)
    health = np.array([[data['you']['health']+1], [data['you']['health']]]).transpose(1, 0)
    map_width = data['board']['width']

    health_dict = make_health_dict(data)

    data = {"state": state.tolist(), "snake_id": snake_id.tolist(), 
        "turn_count": turn.tolist(), "health": health.tolist(),
        "all_health": health_dict, "map_width": map_width, "json": data}
    payload = json.dumps(data)
    response = runtime.invoke_endpoint(EndpointName="battlesnake-endpoint",
                                       ContentType='application/json',
                                       Body=payload)
    direction_index = json.loads(response['Body'].read().decode())

    return direction_index

def remoteInferenceRLib(previous_state, current_state, data):
    map_width = data['board']['width']
    state = np.concatenate((previous_state, current_state), axis=2)
    state = np.expand_dims(state, 0)

    health_dict = make_health_dict(data)

    data = {"state": state.tolist(), "prev_action": -1, 
            "prev_reward": -1, "seq_lens": -1,  
            "all_health": health_dict, "json": data}

    payload = json.dumps(data)
    response = runtime.invoke_endpoint(EndpointName="battlesnake-endpoint",
                                       ContentType='application/json',
                                       Body=payload)
    direction_index = json.loads(response['Body'].read().decode())
    direction_index = direction_index["outputs"]["heuristisc_action"]
    return direction_index

def end():
    return {
        "statusCode": 200
    }
