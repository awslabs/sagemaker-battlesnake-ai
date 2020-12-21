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

# This script build the snake Lambda package
#
# prerequisite: python 3, venv, pip

if [ $# -lt 3 ]; then
    echo "Please provide the solution name as well as the base S3 bucket name and the region to run build script."
    echo "For example: ./build.sh trademarked-solution-name sagemaker-solutions-build us-west-2"
    exit 1
fi

mkdir build

echo
echo " > Start packaging model on Lambda"
echo

SOLUTION_ASSISTANT_FILE_NAME="solution-assistant.zip"
API_FILE_NAME="api.zip"

# Delete file if it exist already
cleanup () {
    if [ -f $1 ]; then
        echo " > $1 exist, delete it"
        echo
        rm -f "./$1"
    fi
}
cleanup build/$SOLUTION_ASSISTANT_FILE_NAME
cleanup build/$API_FILE_NAME

#### Solution assistance creation
mkdir build/solution-assistant
cp -r ./deployment/CloudFormation/solution-assistant/. ./build/solution-assistant/
(cd ./build/solution-assistant && pip install -r requirements.txt -t ./src/site-packages)
find ./build/solution-assistant -name '*.pyc' -delete
(cd ./build/solution-assistant/src && zip -q -r9 ../../$SOLUTION_ASSISTANT_FILE_NAME *)
rm -rf ./build/solution-assistant

# Endpoint API creation
mkdir build/model-lambda-package
cp -r ./deployment/LambdaGateway/. ./build/model-lambda-package/
(cd ./build/model-lambda-package && pip install -r requirements.txt -t ./src/site-packages --platform manylinux1_x86_64 --python-version 3.7 --only-binary :all:)
find ./build/model-lambda-package -name '*.pyc' -delete
(cd ./build/model-lambda-package/src && zip -q -r9 ../../$API_FILE_NAME *)
rm -rf ./build/model-lambda-package

# Upload to S3
s3_prefix="s3://$2-$3/$1"
echo "Using S3 path: $s3_prefix"
aws s3 cp --recursive source $s3_prefix/source --exclude '.*' --exclude "*~"
aws s3 cp --recursive deployment $s3_prefix/deployment --exclude '.*' --exclude "*~"
aws s3 cp --recursive build $s3_prefix/build
aws s3 cp README.md $s3_prefix/

# Copy solution artefacts to the folder
aws s3 cp "s3://sagemaker-solutions-artifacts/$1/model-complete.tar.gz" $s3_prefix/build/model-complete.tar.gz
