#!/bin/bash

set -e

# OVERVIEW
# This script executes an existing Notebook file on the instance during start using nbconvert(https://github.com/jupyter/nbconvert)

# PARAMETERS

ENVIRONMENT=python3
NOTEBOOK_FILE=/home/ec2-user/SageMaker/deployEndpoint.ipynb

source /home/ec2-user/anaconda3/bin/activate "$ENVIRONMENT"

echo "Start endpoint creation" > mylog.log
date >> mylog.log

nohup jupyter nbconvert "$NOTEBOOK_FILE" --ExecutePreprocessor.timeout=600 --ExecutePreprocessor.kernel_name=python --execute&

echo "End endpoint creation" >> mylog.log
date >> mylog.log

source /home/ec2-user/anaconda3/bin/deactivate
