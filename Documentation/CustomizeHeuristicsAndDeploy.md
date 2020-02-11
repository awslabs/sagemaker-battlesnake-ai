# Customize AI behavior

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
