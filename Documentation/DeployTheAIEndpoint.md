## Deploy a pretrained snake AI into your AWS account

This section will deploy a pre-trained AI on a Lambda function. An API Gateway will be created in front of it to provide a snake API entripoint (see [BattleSnake API](https://docs.battlesnake.com/snake-api)).

To deploy this into Canada AWS region please use the link below:

__<a href="https://ca-central-1.console.aws.amazon.com/cloudformation/home?region=ca-central-1#/stacks/create/review?templateURL=https://battlesnake-aws-ca-central-1.s3.ca-central-1.amazonaws.com/cloudformation/deploy-battlesnake-endpoint.yaml&stackName=DemoSagemaker" target="_blank">Deploy the Snake</a>__

_You need to be logged into the AWS account where you want to deploy the stack._

Check all permissions:

![Accept Permissions](https://github.com/awslab/sagemaker-battlesnake-ai/raw/master/Documentation/images/create-stack.png "Permission checkboxes")

Click "Create Stack"

After about a minute, the stack status should be CREATE_COMPLETE:

![Creation complete](https://github.com/awslab/sagemaker-battlesnake-ai/raw/master/Documentation/images/create-complete.png "Creation complete")

Open the outputs tab and click on "Start Method" link to test that the deployment work:

![Output tab](https://github.com/awslab/sagemaker-battlesnake-ai/raw/master/Documentation/images/outputs.png "Output tab")

You should see:

![Successfull result](https://github.com/awslab/sagemaker-battlesnake-ai/raw/master/Documentation/images/working.png "Result")

Again on output tab, the value "Snake URL" is your Snake URL, you can use it on [BattleSnake](https://play.battlesnake.com/) !

Add your snake:

![Add snake](https://github.com/awslab/sagemaker-battlesnake-ai/raw/master/Documentation/images/addsnake.png "Add snake")

Play:

![Battlesnake Board](https://github.com/awslab/sagemaker-battlesnake-ai/raw/master/Documentation/images/game.png "Battlesnake Board")
