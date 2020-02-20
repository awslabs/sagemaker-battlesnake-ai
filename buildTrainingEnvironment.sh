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

if find $1 -mindepth 1 | read > /dev/null; then
   echo "$1 not empty. Please enter an empty directory"
else
    echo "Installing the battlesnake gym"
    cd TrainingEnvironment
    source activate mxnet_p36
    pip install -e .
    cd ..
    
    echo "Create the entry points"
    mkdir $1
    cd $1
    mkdir battlesnake_gym/
    mkdir battlesnake_src
    cd ..
    cp -a TrainingEnvironment/battlesnake_gym $1/battlesnake_gym/battlesnake_gym
    cp -a TrainingEnvironment/setup.py $1/battlesnake_gym/
    cp -a TrainingEnvironment/requirements.txt $1/battlesnake_gym/

    cp -a TrainingEnvironment/examples/. $1/battlesnake_src/
    cp TrainingEnvironment/requirements.txt $1/battlesnake_src/

    echo "Create the notebooks"
    cp TrainingEnvironment/notebooks/* $1
fi

