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

mkdir -p $1

if find $1 -mindepth 1 | read > /dev/null; then
   echo "$1 not empty. Please enter an empty directory"
else
    echo "Create the entry points"
    cd $1
    mkdir battlesnake_gym/
    mkdir mxnet_src
    mkdir mxnet_inference
    mkdir mxnet_inference/src
    mkdir mxnet_inference/pretrained_models
    mkdir rllib_src
    mkdir rllib_common
    mkdir rllib_inference
    mkdir heuristics

    cd ..
    
    echo "Copying battlesnake gym"
    cp -a BattlesnakeGym/. $1/battlesnake_gym/
    
    echo "Copying heuristics"
    cp -a Heuristics/. $1/heuristics/.
    
    echo "Copying MXNet environment"
    cp -a MXNet/MXNetTrainingEnvironment/src/. $1/mxnet_src/
    cp -a MXNet/MXNetTrainingEnvironment/notebooks/. $1
    cp -a MXNet/MXNetInferenceEndpoint/PretrainedModels/. $1/mxnet_inference/pretrained_models
    cp -a MXNet/MXNetInferenceEndpoint/endpoint/. $1/mxnet_inference/src
        
    echo "Copying RLlib environment"
    cp -a RLlib/RLlibTrainingEnvironment/src/. $1/rllib_src/
    cp -a RLlib/RLlibTrainingEnvironment/common/. $1/rllib_common/
    cp -a RLlib/RLlibTrainingEnvironment/RLlibPolicyTraining.ipynb $1
    
    cp -a RLlib/RLLibInferenceEndpoint/. $1/rllib_inference
fi