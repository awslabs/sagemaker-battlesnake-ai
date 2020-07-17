#!/bin/bash
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

set -e

# OVERVIEW
# This script executes an the environement bootstrap notebook 
# This will generate a Sagemaker Model, Endpoint configuration and Endpoint

# PARAMETERS

RL_METHOD=$1
FOLDER=$RL_METHOD"Env"
NOTEBOOK_FILE=/home/ec2-user/SageMaker/battlesnake/$FOLDER/deployEndpoint.ipynb

# Create the SagMaker endpoint
source /home/ec2-user/anaconda3/bin/activate python3
nohup jupyter nbconvert "$NOTEBOOK_FILE" --ExecutePreprocessor.timeout=600 --ExecutePreprocessor.kernel_name=python --execute&

chown -R ec2-user:ec2-user $FOLDER