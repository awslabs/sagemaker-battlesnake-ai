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

def model_fn(model_dir):
    #load pretrained model
    print("model_fn model_dir={} glob={}".format(model_dir, glob.glob("{}/Model/*".format(model_dir))))
    symbol = None
    params = None
    for filename in glob.glob("{}/Model/*".format(model_dir)):
        if "local-symbol.json" in filename:
            symbol = filename
        if "local-0000.params" in filename:
            params = filename
            
    for filename in glob.glob("{}/*".format(model_dir)):
        if "local-symbol.json" in filename:
            symbol = filename
        if "local-0000.params" in filename:
            params = filename

    model = gluon.SymbolBlock.imports(
        symbol, ['data0', 'data1', 'data2', 'data3'],
        params) 
    print("model_fn symbol {} params {}".format(symbol, params))
    return model
    
def transform_fn(model, data, content_type, output_content_type):
    """
    Transform incoming requests.
    """
    #check if GPUs area available
    ctx = mx.gpu() if mx.context.num_gpus() > 0 else mx.cpu()
    
    data = json.loads(data)
    
    #convert input data into MXNet NDArray
    state = mx.nd.array(data["state"], ctx=ctx)
    snake_id = mx.nd.array(data["snake_id"], ctx=ctx)
    turn_count = mx.nd.array(data["turn_count"], ctx=ctx)
    snake_health = mx.nd.array(data["health"], ctx=ctx)
    
    #inference
    action = model(state, snake_id, turn_count, snake_health)
    action = action.asnumpy()[0]
    
    heuristics_state = np.array(data["state"])
    heuristics_id = np.array(data["snake_id"])
    heuristics_turn = np.array(data["turn_count"])
    heuristics_health = np.array(data["health"])
      
    converted_action = heuristics.run(heuristics_state, 
                                      heuristics_id,
                                      heuristics_turn+1,
                                      heuristics_health, 
                                      action=action)
    output = converted_action.tolist()
    
    #decode result as json string
    response_body = json.dumps(output)
    
    return response_body, output_content_type
