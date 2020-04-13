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

if [ -z "$1" ]; then
    echo "Please enter a directory name as an argument"
    exit 1
fi

if [ -z "$2" ]; then
    echo "Please a training environment choices: [MXNet, RLlib]"
    exit 1
elif [ "$2" != "MXNet" ] && [ "$2" != "RLlib" ]; then
    echo "Please a valid training environment choices: [MXNet, RLlib]"
    exit 1
fi

mkdir -p $1

if find $1 -mindepth 1 | read > /dev/null; then
   echo "$1 not empty. Please enter an empty directory"
else
    echo "Create the entry points"
    cd $1
    mkdir battlesnake_gym/
    if [ $2 == "MXNet" ]; then

        mkdir mxnet_src
        mkdir mxnet_inference
        mkdir mxnet_inference/src
        mkdir mxnet_inference/pretrained_models
    
    else
        mkdir rllib_src
        mkdir rllib_common
        mkdir rllib_inference
        mkdir rllib_inference/src
    fi

    cd ..
    
    echo "Copying battlesnake gym"
    cp -a BattlesnakeGym/. $1/battlesnake_gym/

    if [ $2 == "MXNet" ]; then
        echo "Copying MXNet environment"
        cp -a MXNet/TrainingEnvironment/src/. $1/mxnet_src/
        cp -a MXNet/TrainingEnvironment/PolicyTraining.ipynb $1
        cp -a MXNet/InferenceEndpoint/PretrainedModels/. $1/mxnet_inference/pretrained_models
        cp -a MXNet/InferenceEndpoint/endpoint/. $1/mxnet_inference/src
        cp -a MXNet/InferenceEndpoint/deployEndpoint.ipynb $1/deployEndpoint.ipynb
        
        cp -a MXNet/HeuristicsDevelopment/heuristics_utils.py $1
        cp -a MXNet/HeuristicsDevelopment/HeuristicsDeveloper.ipynb $1
        cp -a Heuristics/battlesnake_heuristics.py $1/mxnet_inference/src/battlesnake_heuristics.py
    else
        echo "Copying RLlib environment"
        cp -a RLlib/TrainingEnvironment/src/. $1/rllib_src/
        cp -a RLlib/TrainingEnvironment/common/. $1/rllib_common/
        cp -a RLlib/TrainingEnvironment/PolicyTraining.ipynb $1
        cp -a RLlib/InferenceEndpoint/endpoint/. $1/rllib_inference/src
        cp -a RLlib/InferenceEndpoint/model.tar.gz $1/rllib_inference/.
        cp -a RLlib/InferenceEndpoint/deployEndpoint.ipynb $1/deployEndpoint.ipynb

        cp -a RLlib/HeuristicsDevelopment/heuristics_utils.py $1
        cp -a RLlib/HeuristicsDevelopment/HeuristicsDeveloper.ipynb $1
        cp -a Heuristics/battlesnake_heuristics.py $1/rllib_inference/src/battlesnake_heuristics.py
    fi
fi