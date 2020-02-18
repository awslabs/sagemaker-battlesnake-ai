if find $1 -mindepth 1 | read > /dev/null; then
   echo "$1 not empty. Please enter an empty directory"
else
    echo "Installing the battlesnake gym"
    cd TrainingEnvironment
    source activate mxnet_p36
    pip install -e .
    cd ..
    
    echo "Create the entry points"
    mkdir $1
    cd $1
    mkdir battlesnake_gym/
    mkdir battlesnake_src
    cd ..
    cp -a TrainingEnvironment/battlesnake_gym $1/battlesnake_gym/battlesnake_gym
    cp -a TrainingEnvironment/setup.py $1/battlesnake_gym/
    cp -a TrainingEnvironment/requirements.txt $1/battlesnake_gym/

    cp -a TrainingEnvironment/examples/. $1/battlesnake_src/
    cp TrainingEnvironment/requirements.txt $1/battlesnake_src/

    echo "Create the notebooks"
    cp TrainingEnvironment/notebooks/* $1
fi

