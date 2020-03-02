# STEP 1 - Deploy a pretrained AI

This section will deploy a pre-trained AI on your AWS account. This AI will expose the [BattleSnake API](https://docs.battlesnake.com/snake-api).

## Architecture

The architecture deployed will be:

![Pretrained Architecture](images/ArchitectureSagemakerBattleSnake.png "Pretrained Architecture")

> __Estimated cost__ : If you run this architecture for a limited time, it will fit in the [AWS Free tiers](https://aws.amazon.com/free/). The AWS Free tiers can support 500 000 snake API invokation on a 125 hours period per month on the first two months. The free tiers also include 250 hours per month of this notebook instance on the first two months.
> After the free tiers the charge will be $0.269 per hour ($6.5 per 24 hour period) for the endpoint instance, $0.269 per hour for the notebook instance and $6 per million Snake API call.
> __Saving tip__ : Once you have finished working (respectively participating to games) you can stop your notebook instance (resp. your endpoint instance) to stop consuming free tiers or occuring charge. You can restart them later to continue your work.
> See pricing details: [Amazon Sagemaker pricing](https://aws.amazon.com/sagemaker/pricing/), [AWS Lambda pricing](https://aws.amazon.com/lambda/pricing/), [Amazon API Gateway pricing](https://aws.amazon.com/api-gateway/pricing/)

## Deploy environment

Use the links below to deploy the model in the region you like*:

| Region        | deployment link |
| ------------- | :-------------:|
| __US West (Oregon) (us-west-2)__**      | [deploy](https://us-west-2.console.aws.amazon.com/cloudformation/home?region=us-west-2#/stacks/create/review?templateURL=https://battlesnake-aws-us-west-2.s3.us-west-2.amazonaws.com/cloudformation/deploy-battlesnake-endpoint.yaml&stackName=BattleSnakeEnvironment) |
| US East (N. Virginia) us-east-1     | [deploy](https://us-east-1.console.aws.amazon.com/cloudformation/home?region=us-east-1#/stacks/create/review?templateURL=https://battlesnake-aws-us-east-1.s3.us-east-1.amazonaws.com/cloudformation/deploy-battlesnake-endpoint.yaml&stackName=BattleSnakeEnvironment) |
| Canada (Central) ca-central-1     | [deploy](https://ca-central-1.console.aws.amazon.com/cloudformation/home?region=ca-central-1#/stacks/create/review?templateURL=https://battlesnake-aws-ca-central-1.s3.ca-central-1.amazonaws.com/cloudformation/deploy-battlesnake-endpoint.yaml&stackName=BattleSnakeEnvironment) |
| South America (São Paulo) sa-east-1     | [deploy](https://sa-east-1.console.aws.amazon.com/cloudformation/home?region=sa-east-1#/stacks/create/review?templateURL=https://battlesnake-aws-sa-east-1.s3.sa-east-1.amazonaws.com/cloudformation/deploy-battlesnake-endpoint.yaml&stackName=BattleSnakeEnvironment) |
| Europe (Ireland) eu-west-1     | [deploy](https://eu-west-1.console.aws.amazon.com/cloudformation/home?region=eu-west-1#/stacks/create/review?templateURL=https://battlesnake-aws-eu-west-1.s3.eu-west-1.amazonaws.com/cloudformation/deploy-battlesnake-endpoint.yaml&stackName=BattleSnakeEnvironment) |
| Europe (Paris) eu-west-3     | [deploy](https://eu-west-3.console.aws.amazon.com/cloudformation/home?region=eu-west-3#/stacks/create/review?templateURL=https://battlesnake-aws-eu-west-3.s3.eu-west-3.amazonaws.com/cloudformation/deploy-battlesnake-endpoint.yaml&stackName=BattleSnakeEnvironment) |
| Asia Pacific (Sydney) ap-southeast-2    | [deploy](https://ap-southeast-2.console.aws.amazon.com/cloudformation/home?region=ap-southeast-2#/stacks/create/review?templateURL=https://battlesnake-aws-ap-southeast-2.s3.ap-southeast-2.amazonaws.com/cloudformation/deploy-battlesnake-endpoint.yaml&stackName=BattleSnakeEnvironment) |

_* You need to be logged into the AWS account where you want to deploy the stack._

_** the official BattleSnake platform run in us-west-2, selecting this one will provide you the lowest latency_

Customize your snake appearance (color, head, tail) or leave default and scroll down at the end of the page. Then check all permissions:

![Accept Permissions](images/create-stack.png "Permission checkboxes")

Click "Create Stack"

After about 10 minutes, the stack status should be CREATE_COMPLETE:

![Creation complete](images/create-complete.png "Creation complete")

Open the outputs tab and click on "CheckSnakeStatus" link to see if the Snake is ready:

![Output tab](images/outputs.png "Output tab")

After about 10 minutes you should see somthing like this:

![Successfull result](images/working.png "Result")

> __Troubleshooting__ : If after 20 minutes the snake is not ready, you can go on Amazon Sagemaker in the AWS console and look for Inference Endpoint. If you don't see any, go to Notebook instance, clic on your instance, scroll down and clic on View Logs. Then clic on BattleSnakeNotebook/LifecycleConfigOnStart and see if you find any error.

> If the error is _The account-level service limit 'ml.m5.xlarge for endpoint usage' is 0 Instances_ then delete the stack, and recreate it selecting a different instance type on the AWS CloudFormation stack parameter page.

Again on output tab, the value "Snake URL" is your Snake URL, you can use it on [BattleSnake](https://play.battlesnake.com/) !

Add your snake on the [Battlesnake platform](https://play.battlesnake.com/) and copy your URL:

![Add snake](images/addsnake.png "Add snake")

Create a game, select your snake by his name, add opponents and start the game:

![Battlesnake Board](images/game.png "Battlesnake Board")

## Stop instances

Once you finish your work for the day or you finish participating to games with your snake you can stop instances.

To stop the notebook instance (dev environment) go to Amazon Sagemaker in the AWS console, navigate to notebook instances, select your instance, clic action, and choose stop.

To stop the endpoint instance (dev environment) go to Amazon Sagemaker in the AWS console, navigate to inference endpoint instances, select your instance, clic action, and choose stop.

## Next step: customize snake

Go to __[STEP 2 - Customize the AI heuristics](UpdateHeuristicsAndDeploy.md)__ to start making change to the snake behavior.

## Cleanup environment

To cleanup the environment go to AWS CloudFormation, select your BattleSnakeEnvironment stack and click delete.

__WARNING : Delete stack will erase the BattleSnake environment data, logs and code change.__ Make sure you saved your work before doing this.
