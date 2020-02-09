# deployableCFSagemaker
Simple CF demo to deploy Sagemaker stuff


To deploy this into Canada AWS region please use the link below:

__<a href="https://ca-central-1.console.aws.amazon.com/cloudformation/home?region=ca-central-1#/stacks/create/review?templateURL=https://yvr-immersion-days.s3.ca-central-1.amazonaws.com/cloudformation/demo-sagemaker.yaml&stackName=DemoSagemaker">Deploy the Sagemaker demo</a>__

_You need to be logged into the AWS account where you want to deploy the stack._

Check all permissions:


Click "Create Stack":


After about a minute, the stack status should be CREATE_COMPLETE:


Open the outputs tab and click on "" to test that the deployment work:

You should see:


Again on output tab, the value "Snake URL" is your Snake URL, you can use it on BattleSnake.io !!!

## Modify the Snake

There is two type of change you can do against the Snake:

- Add more heuristics
- Update the AI model

### Add more heuristics

### Update the AI model

_...coming soon_

## Package Developers

The package have been build like that ...

