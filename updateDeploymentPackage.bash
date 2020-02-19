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
        AWS_PROFILE="--profile "$2
        echo " using AWS profile "$2
fi

echo

for ix in ${!S3_REGIONS[*]}
do
    echo " > Copy cloudformation to region "${S3_REGIONS[$ix]}
    echo 
    COMMAND="aws s3 cp CloudFormation/deploy-battlesnake-endpoint.yaml s3://$S3_PREFIX${S3_REGIONS[$ix]}/cloudformation/deploy-battlesnake-endpoint.yaml $AWS_PROFILE"
    echo $COMMAND
    eval $COMMAND
    echo
    echo " > Copy lambda model inference package to region "${S3_REGIONS[$ix]}
    echo
    COMMAND="aws s3 cp model-lambda-package.zip s3://$S3_PREFIX${S3_REGIONS[$ix]}/lambda/model-lambda-package.zip $AWS_PROFILE"
    echo $COMMAND
    eval $COMMAND
    echo
done

