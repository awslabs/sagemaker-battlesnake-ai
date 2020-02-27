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

echo
echo " > Start packaging model on Lambda"
echo

PACKAGE_FILE_NAME="model-lambda-package.zip"
DEPLOY_PACKAGE_FILE_NAME="deployment-lambda-package.zip"
LAYER_PACKAGE_FILE_NAME="mxnet-layer-package.zip"

# Delete file if it exist already
cleanup () {
    if [ -f $1 ]; then
        echo " > $1 exist, delete it"
        echo
        rm -f "./$1"
    fi
}
cleanup $PACKAGE_FILE_NAME
cleanup $LAYER_PACKAGE_FILE_NAME
cleanup $DEPLOY_PACKAGE_FILE_NAME

# Create the deployment Lambda package
cd CloudFormation
zip -rq ../$DEPLOY_PACKAGE_FILE_NAME lambda.py
cd ..

# Create the subfolder python
mkdir -p packageLayerTmp/python

# Create a virtualenv python 3
python3 -m venv venv
source venv/bin/activate

# Install mxnet package
python3 -m pip install mxnet

# Copy model into the package directory
cd venv/lib/python3*/site-packages
# remove unused file to remain under 250Mb (max Lambda package size)
rm -rf *.dist-info
rm -rf numpy pip setuptools easy_install*

mv * ../../../../packageLayerTmp/python
cd ../../../../packageLayerTmp

# display package content for debug
echo
echo " > The lambda layer package will contain:"
echo

ls python

echo

# zip it into a lambda package
zip -rq ../$LAYER_PACKAGE_FILE_NAME .

cd ..

# Deactivate virtualenv
deactivate

# Cleanup
rm -rf venv
rm -rf packageLayerTmp

echo
echo " > Your Lambda Layer package $LAYER_PACKAGE_FILE_NAME is ready"
echo

TMP_BUILD_FOLDER=buildLambaTmpFolder

mkdir $TMP_BUILD_FOLDER
cd $TMP_BUILD_FOLDER
cp -r ../InferenceEndpoint/SnakeInference/* .

# display package content for debug
echo
echo " > The lambda package will contain:"
echo

find .

# zip it into a lambda package
zip -q9r ../$PACKAGE_FILE_NAME .

echo

# Cleanup temp file
cd ..
rm -rf $TMP_BUILD_FOLDER

echo
echo " > Your Lambda package $PACKAGE_FILE_NAME is ready"
echo
