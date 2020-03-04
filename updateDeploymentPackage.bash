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
DEV_REGION="ap-northeast-2"

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
        S3_REGIONS=("ca-central-1" "us-west-2" "us-east-1" "sa-east-1" "eu-west-1" "eu-west-3" "ap-southeast-2")
        echo " deploying to all "${#S3_REGIONS[*]}" regions"
    else
        if [ "$1" == "dev" ]
            then
                S3_REGIONS=$DEV_REGION
            else
                S3_REGIONS=$1
        fi
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
    PYTHON_VERSION="python3.7"
    LAYER_NAME="AWSLambda-Python37-MXNet"
    LAYER_PACKAGE_SOURCE_KEY="lambda/mxnet-layer-package.zip"
    TEMP_FILE="cmddumptemp.txt"

    COMMAND="aws s3 cp mxnet-layer-package.zip s3://$S3_PREFIX$1/$LAYER_PACKAGE_SOURCE_KEY --region $1 $AWS_PROFILE"
    displayEndExecCmd \${COMMAND} " > Copy MXNet lambda layer package source to region "$1

    LAYER_DESCRIPTION="MXnet layer for $PYTHON_VERSION. This layer require SciPy 3.7 layer"
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
    CF_TEMPLATE_FILE="CloudFormation/deploy-battlesnake-endpoint.yaml"
    if [ "$1" == "dev" ]
        then
            echo
            echo "DEVELOPEMENT > Replacing github source repo by dev one"
            echo
            sed -e "s/https\:\/\/github\.com\/awslabs\/sagemaker-battlesnake-ai\.git/https\:\/\/github\.com\/JohnyFicient\/randomproject\.git/g" CloudFormation/deploy-battlesnake-endpoint.yaml> dev.yaml
            CF_TEMPLATE_FILE=dev.yaml
    fi
    COMMAND="aws s3 cp $CF_TEMPLATE_FILE s3://$S3_PREFIX${S3_REGIONS[$ix]}/cloudformation/deploy-battlesnake-endpoint.yaml $AWS_PROFILE"
    displayEndExecCmd \${COMMAND} " > Copy cloudformation scripts to region "${S3_REGIONS[$ix]}

    # WARNING:
    # Every time you update the Lambda Layer the version increase by one.
    # So you'll need to update the Cloudformation script with that new number.
    # Decomment this line only when you want to update the layer
    #
    # updateOrCreateLayer ${S3_REGIONS[$ix]}

    COMMAND="aws s3 cp deployment-lambda-package.zip s3://$S3_PREFIX${S3_REGIONS[$ix]}/lambda/deployment-lambda-package.zip $AWS_PROFILE"
    displayEndExecCmd \${COMMAND} " > Copy deployment lambda package to region "${S3_REGIONS[$ix]}

    COMMAND="aws s3 cp model-lambda-package.zip s3://$S3_PREFIX${S3_REGIONS[$ix]}/lambda/model-lambda-package.zip $AWS_PROFILE"
    displayEndExecCmd \${COMMAND} " > Copy lambda model inference package to region "${S3_REGIONS[$ix]}
done

if [ "$1" == "dev" ]
    then
    echo
    echo "DEVELOPEMENT > This is dev env : Go to this URL to deploy"
    echo "https://$DEV_REGION.console.aws.amazon.com/cloudformation/home?region=$DEV_REGION#/stacks/create/review?templateURL=https://battlesnake-aws-$DEV_REGION.s3.$DEV_REGION.amazonaws.com/cloudformation/deploy-battlesnake-endpoint.yaml&stackName=BattlesnakeEnvironment"
    echo
fi

