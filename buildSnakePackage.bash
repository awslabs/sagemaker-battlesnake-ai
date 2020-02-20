# This script build the snake Lambda package
#
# prerequisite: python 2.7, virtualenv, pip

echo " "
echo " > Start packaging model on Lambda"
echo " "

# Create a virtualenv python 2.7
virtualenv venv
source venv/bin/activate

# Install mxnet package
pip install mxnet

# Copy model into the package directory
cp -r PreTrainedSnake/SnakeModelEndpoint/* venv/lib/python2.7/site-packages
cd venv/lib/python2.7/site-packages
# remove unused file to remain under 250Mb (max Lambda package size)
rm -rf *.dist-info
rm -rf numpy pip setuptools

# zip it into a lambda package
zip -9r ../../../../model-lambda-package.zip .

# Deactivate virtualenv
deactivate

# Cleanup
rm -rf venv

echo " "
echo " > Your Lambda package model-lambda-package.zip is ready"
echo " "
