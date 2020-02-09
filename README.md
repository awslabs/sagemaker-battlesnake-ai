# deployableCFSagemaker

This project shows how to build and deploy an AI for the game [BattleSnake](https://play.battlesnake.com/) with [AWS Machine Learning](https://aws.amazon.com/machine-learning/)!

## Deploy a pretrained snake AI into your AWS account

This section will deploy a pre-trained AI on a Lambda function. An API Gateway will be created in front of it to provide a snake API entripoint (see [BattleSnake API](https://docs.battlesnake.com/snake-api)).

To deploy this into Canada AWS region please use the link below:

__<a href="https://ca-central-1.console.aws.amazon.com/cloudformation/home?region=ca-central-1#/stacks/create/review?templateURL=https://yvr-immersion-days.s3.ca-central-1.amazonaws.com/cloudformation/demo-sagemaker.yaml&stackName=DemoSagemaker" target="_blank">Deploy the Snake</a>__

_You need to be logged into the AWS account where you want to deploy the stack._

Check all permissions:

![Accept Permissions](https://github.com/xavierraffin/deployableCFSagemaker/raw/master/Documentation/images/create-stack.png "Permission checkboxes")

Click "Create Stack"

After about a minute, the stack status should be CREATE_COMPLETE:

![Creation complete](https://github.com/xavierraffin/deployableCFSagemaker/raw/master/Documentation/images/create-complete.png "Creation complete")

Open the outputs tab and click on "Start Method" link to test that the deployment work:

![Output tab](https://github.com/xavierraffin/deployableCFSagemaker/raw/master/Documentation/images/outputs.png "Output tab")

You should see:

![Successfull result](https://github.com/xavierraffin/deployableCFSagemaker/raw/master/Documentation/images/working.png "Result")

Again on output tab, the value "Snake URL" is your Snake URL, you can use it on [BattleSnake](https://play.battlesnake.com/) !

Add your snake:

![Add snake](https://github.com/xavierraffin/deployableCFSagemaker/raw/master/Documentation/images/addsnake.png "Add snake")

Play:

![Battlesnake Board](https://github.com/xavierraffin/deployableCFSagemaker/raw/master/Documentation/images/game.png "Battlesnake Board")

## Modify the Snake

There is two type of change you can do against the Snake:

- __Add more heuristics__ : keep the model but add additionnal hard-wire decision
- __Update the AI model__ : train the AI again with different settings

Next sections explain how to do each of these two type of changes.

### Add more heuristics

Here we keep the AI as is but we add some code to change the AI movement decision.

For example, you can calculate if the move will make you collide into a snake body or head with a longer body (in both case you die).

Another one will be detect that you may be able to kill another shorter snake colliding head to head.

_If you do clever things, your pull request is welcome!_

__STEP 1: Edit the source code__

Inside SnakeLambdaPackage/lambda.py on the move function add your code.

__STEP 2: Deploy the new Snake__

Generate a lambda package like this:

```
cd SnakeLambdaPackage
zip -9r lambda.zip  .
```

__STEP 3: Deploy the new Snake__

Push this code to an S3 bucket of you using the console or the CLI:

```
aws s3 cp lambda.zip s3://<YOUR-S3-BUCKET>/lambda.zip
```

Deploy the Lambda using CloudFormation: (you have to do that in the same region where your bucket is).

```
aws cloudformation create-stack --stack-name <YOUR-STACK-NAME> --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM CAPABILITY_AUTO_EXPAND --template-body file://improved-ai.yaml --parameters ParameterKey=S3Bucket,ParameterValue=<YOUR-S3-BUCKET> ParameterKey=S3Key,ParameterValue=lambda.zip
```

_obviously you can deploy the lambda package manually without cloudformation. But in that case you'll have to create the API Gateway too and you will need to add the NumPy Lambda layer to make it working._

Once the CloudFormation stack deployed, get the Snake API URL in the output section.


### Update the AI model

_...coming soon_

## Package Developers

Here are instructions that have been used to create the initial Lambda package.

On a Amazon Linux 1 machine: (SnakeSource contains the python code of the AI)

```
virtualenv venv
source venv/bin/activate
pip install mxnet

cp SnakeSource/* venv/lib/python2.7/site-packages

cd venv/lib/python2.7/site-packages && zip -r ../../../../lambda.zip . && cd -
```

Publish the Lambda package on a public S3 bucket:

```
aws s3 cp lambda.zip s3://yvr-immersion-days/cloudformation/
```

For the Cloudformation script same command:

```
aws s3 cp demo-sagemaker.yaml s3://yvr-immersion-days/cloudformation/demo-sagemaker.yaml
```

