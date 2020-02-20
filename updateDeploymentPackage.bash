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

# This script is for AWS maintainer only
# It deploys all CloudFormation scripts and packages to the project S3 buckets
# It will fail for user without AWS credentials

S3_PREFIX="battlesnake-aws-"

echo

if [ $# -eq 0 ]
  then
    echo "No arguments supplied"
    echo 
    echo " usage:"
    echo "  ./updateDeploymentPackage.bash all|us-west-2|ca-central-1|... [profileName]"
    echo
    exit
fi

echo " > Start copying packaging process"
echo

if [ "$1" == "all" ]
    then
        S3_REGIONS=("ca-central-1" "us-west-2" "us-east-1" "sa-east-1" "eu-west-1" "eu-west-3" "ap-northeast-2" "ap-southeast-2")
        echo " deploying to all "${#S3_REGIONS[*]}" regions"
    else
        S3_REGIONS=$1
        echo " deploying to region "$S3_REGIONS
fi

echo

if [ -z "$2" ]
    then
        AWS_PROFILE=""
        echo " no profile supplied, using default AWS credentials"
    else
        AWS_PROFILE="--profile $2"
        echo " using AWS profile "$2
fi

echo

displayEndExecCmd () {
  echo "$2"
  echo
  eval cmd="$1"
  echo ${cmd}
  eval ${cmd}
  echo
}

updateOrCreateLayer () {
    PYTHON_VERSION="python2.7"
    LAYER_NAME="AWSLambda-Python27-MXNet"
    LAYER_PACKAGE_SOURCE_KEY="lambda/mxnet-layer-package.zip"
    TEMP_FILE="cmddumptemp.txt"

    COMMAND="aws s3 cp mxnet-layer-package.zip s3://$S3_PREFIX$1/$LAYER_PACKAGE_SOURCE_KEY --region $1 $AWS_PROFILE"
    displayEndExecCmd \${COMMAND} " > Copy MXNet lambda layer package source to region "$1

    LAYER_DESCRIPTION="MXnet layer for $PYTHON_VERSION require SciPy layer"
    COMMAND="aws lambda publish-layer-version --layer-name $LAYER_NAME --description \"$LAYER_DESCRIPTION\" --license-info \"MIT\" --content S3Bucket=$S3_PREFIX$1,S3Key=$LAYER_PACKAGE_SOURCE_KEY --compatible-runtimes $PYTHON_VERSION --region $1 $AWS_PROFILE > $TEMP_FILE"
    displayEndExecCmd \${COMMAND} " > Create Lambda layer in region "$1

    LAYER_VERSION=`cat $TEMP_FILE | grep "Version\":" | sed 's/[^0-9]*//g'`
    rm -f $TEMP_FILE
    echo "Layer Version is $LAYER_VERSION"

    COMMAND="aws lambda add-layer-version-permission --layer-name $LAYER_NAME --statement-id public --action lambda:GetLayerVersion  --principal \"*\" --version-number $LAYER_VERSION --output text --region $1 $AWS_PROFILE"
    displayEndExecCmd \${COMMAND} " > Making Layer Version public in region "$1
}

for ix in ${!S3_REGIONS[*]}
do
    COMMAND="aws s3 cp CloudFormation/deploy-battlesnake-endpoint.yaml s3://$S3_PREFIX${S3_REGIONS[$ix]}/cloudformation/deploy-battlesnake-endpoint.yaml $AWS_PROFILE"
    displayEndExecCmd \${COMMAND} " > Copy cloudformation scripts to region "${S3_REGIONS[$ix]}
    
   # WARNING:
   # Every time you update the Lambda Layer the version increase by one.
   # So you'll need to update the Cloudformation script with that new number.
   # Decomment this line only when you want to update the layer
    #
   # updateOrCreateLayer ${S3_REGIONS[$ix]}
    
    COMMAND="aws s3 cp model-lambda-package.zip s3://$S3_PREFIX${S3_REGIONS[$ix]}/lambda/model-lambda-package.zip $AWS_PROFILE"
    displayEndExecCmd \${COMMAND} " > Copy lambda model inference package to region "${S3_REGIONS[$ix]}
done

