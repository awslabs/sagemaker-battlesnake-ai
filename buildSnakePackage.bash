# This script build the snake Lambda package
#
# prerequisite: python 2.7, virtualenv, pip

echo
echo " > Start packaging model on Lambda"
echo

PACKAGE_FILE_NAME="model-lambda-package.zip"
LAYER_PACKAGE_FILE_NAME="mxnet-layer-package.zip"

# Delete file if it exist already
cleanup () {
    if [ -f $1 ]; then
        echo " > $1 exist, delete it"
        echo
        rm -f "./$1"
    fi
}
cleanup $PACKAGE_FILE_NAME
cleanup $LAYER_PACKAGE_FILE_NAME

# Create a virtualenv python 2.7
virtualenv venv
source venv/bin/activate

# Install mxnet package
pip install mxnet

# Copy model into the package directory
cd venv/lib/python2.7/site-packages
# remove unused file to remain under 250Mb (max Lambda package size)
rm -rf *.dist-info
rm -rf numpy pip setuptools easy_install*

# display package content for debug
echo
echo " > The lambda layer package will contain:"
echo

ls

echo

# zip it into a lambda package
zip -q9r ../../../../$LAYER_PACKAGE_FILE_NAME .

cd -

# Deactivate virtualenv
deactivate

# Cleanup
rm -rf venv

echo
echo " > Your Lambda Layer package $LAYER_PACKAGE_FILE_NAME is ready"
echo

cd PreTrainedSnake/SnakeModelEndpoint
# zip it into a lambda package
zip -q9r ../../$PACKAGE_FILE_NAME .

# display package content for debug
echo
echo " > The lambda package will contain:"
echo

ls

echo

cd -

echo
echo " > Your Lambda package $PACKAGE_FILE_NAME is ready"
echo