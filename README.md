# Amazon SageMaker for Battlesnake AI

This project shows how to build and deploy an AI for the platform [Battlesnake](https://play.battlesnake.com/) on AWS with [Amazon Sagemaker](https://aws.amazon.com/sagemaker/)!

It is ready to deploy and contains learning materials for AI enthusiast.

__What is Battlesnake?__ (taken from [battlesnake.com](https://docs.battlesnake.com/rules)):

> Battlesnake is an autonomous survival game where your snake competes with others to find and eat food without being eliminated. To accomplish this, you will have to teach your snake to navigate the serpentine paths created by walls, other snakes, and their own growing tail without running out of energy.

## Intention

This project contains ready to use AI for Battlesnake as well as development environment to modify the AI.
The AI takes movement decision in two steps:
 * __1. Run a Neural Network Model__ 
 * __2. Run Heuristics__ some additional code that can override the Model decision (don't colide a wall, eat food if this is safe, ...)

Several pre-trained neural network models are provided within this project as well as some default heuristics. These pre-trained models (snakes) are not designed to win the Battlesnake competition, so you'll have to improve them to have a chance. The training environment is provided for you to make modifications and do retraining to obtain better results.

## Project Organisation

This project can be used in three steps:

- __[STEP 1 - Deploy the environment](Documentation/DeployTheAIEndpoint.md)__ : Deploy a Snake AI in a single click! You are ready to participate to a Battlesnake game.
- __[STEP 2 - Customize the AI heuristics](Documentation/UpdateHeuristicsAndDeploy.md)__ : Customize AI behaviour, visualize your results and publish an upgraded version!
- __[STEP 3 - Train the AI model with your own settings](Documentation/TrainModelAndDeploy.md)__ : The most challenging one: train the AI again with different settings, visualize your result and publish an upgraded version!

_You have to do STEP 1 in order to be able to do STEP 2 or STEP 3._

### Architecture

If you use STEP 1, STEP 2 and STEP 3, you will have the following deployed:

![General Architecture](Documentation/images/ArchitectureSagemakerBattlesnakeFull.png "General Architecture")

### Cost

This project have been design to fit inside the AWS free tiers for some time.
See each steps for duration and cost.

### Content

The source code of the project is organized as:

```
CloudFormation                    # contains the templates and scripts to automate deployment
InferenceEndpoint                 # contains the code of the Snake Endpoint
  > PretrainedModels              # models already trained
  > SnakeInference                # code that expose the Snake API
  > SageMakerEndpoint             # code that is used for model inference
TrainingEnvironment
 > battlesnake_gym                # OpenAI Gym environment that simulate the Battlesnake game engine
 > notebook
   > HeuristicsDeveloper.ipynb    # Jupyter notebook for heuristics development
   > SagemakerModelTraining.ipynb # Jupyter notebook for model training    
```

## License

This project is licensed under the Apache-2.0 License.
