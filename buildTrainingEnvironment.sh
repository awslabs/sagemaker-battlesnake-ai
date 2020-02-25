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

if find $1 -mindepth 1 | read > /dev/null; then
   echo "$1 not empty. Please enter an empty directory"
else
    echo "Installing the battlesnake gym"
    cd TrainingEnvironment
    source activate mxnet_p38
    pip install -e .
    cd ..
    
    echo "Create the entry points"
    mkdir $1
    cd $1
    mkdir battlesnake_gym/
    mkdir battlesnake_src
    mkdir battlesnake_inference
    cd ..
    cp -a TrainingEnvironment/battlesnake_gym $1/battlesnake_gym/battlesnake_gym
    cp -a TrainingEnvironment/setup.py $1/battlesnake_gym/
    cp -a TrainingEnvironment/requirements.txt $1/battlesnake_gym/

    cp -a TrainingEnvironment/examples/. $1/battlesnake_src/
    cp TrainingEnvironment/requirements.txt $1/battlesnake_src/

    echo "Create the notebooks"
    cp TrainingEnvironment/notebooks/* $1
    
    echo "Create the heuristics code"
    cp InferenceEndpoint/SnakeInference/predict.py $1/battlesnake_inference/
    cp InferenceEndpoint/SnakeInference/battlesnake_heuristics.py $1/battlesnake_inference/
    
    echo "Copy the pretrained models"
    cd $1
    mkdir pretrained_models
    cd ..
    cp -r InferenceEndpoint/PretrainedModels/* $1/pretrained_models/
fi

