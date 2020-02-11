
# Package Developers Doc

Here are instructions for developers of this package.

## Create the initial Lambda package.

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

