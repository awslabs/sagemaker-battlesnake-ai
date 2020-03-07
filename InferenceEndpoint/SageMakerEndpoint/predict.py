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

import os
import json
import numpy as np
import mxnet as mx
from mxnet import gluon 
import glob

from battlesnake_heuristics import MyBattlesnakeHeuristics

heuristics = MyBattlesnakeHeuristics()

model_names = ["Model-11x11", 
               "Model-15x15",
               "Model-19x19", 
               "Model-7x7"]

def model_fn(model_dir):
    print("model_fn model_dir={} glob={}".format(model_dir, glob.glob("{}/*".format(model_dir))))
    
    models = {}
    for model_name in model_names:
        symbol_name = "{}/Models/{}/local-symbol.json".format(model_dir, model_name)
        params_name = "{}/Models/{}/local-0000.params".format(model_dir, model_name)
    
        model = gluon.SymbolBlock.imports(
            symbol_name, ['data0', 'data1', 'data2', 'data3'],
            params_name) 
        print("model_fn {} symbol={} params={}".format(model_name, symbol_name, params_name))
        models[model_name] = model
    return models
    
def transform_fn(models, data, content_type, output_content_type):
    """
    Transform incoming requests.
    """
    #check if GPUs area available
    ctx = mx.gpu() if mx.context.num_gpus() > 0 else mx.cpu()
    
    data = json.loads(data)
    
    model_name = "Model-{}x{}".format(data["map_width"], data["map_width"])
    
    #convert input data into MXNet NDArray
    state = mx.nd.array(data["state"], ctx=ctx)
    snake_id = mx.nd.array(data["snake_id"], ctx=ctx)
    turn_count = mx.nd.array(data["turn_count"], ctx=ctx)
    snake_health = mx.nd.array(data["health"], ctx=ctx)
    
    print("running model")
    #inference
    model = models[model_name]
        
    action = model(state, snake_id, turn_count, snake_health)
    action = action.asnumpy()[0]
    
    print("Action is {}".format(action))
    
    heuristics_state = np.array(data["state"])[0, 1, :].transpose(1, 2, 0)
    heuristics_id = np.array(data["snake_id"])[0, 1]
    heuristics_turn = np.array(data["turn_count"])[0, 1]
    heuristics_health = data["all_health"]
    
    converted_heuristic_health = {}
    for k in heuristics_health:
        converted_heuristic_health[int(k)] = heuristics_health[k]
    print("Heuristisc health {}".format(converted_heuristic_health))

    print("state {}".format(heuristics_state.shape))
    print("data json {}".format(data["json"]))
    converted_action, _ = heuristics.run(heuristics_state, 
                                      int(heuristics_id),
                                      int(heuristics_turn)+1,
                                      converted_heuristic_health, 
                                      json=data["json"],
                                      action=action)
    print("converted_action {}".format(converted_action))
    output = converted_action
    
    #decode result as json string
    response_body = json.dumps(output)
    
    return response_body, output_content_type
