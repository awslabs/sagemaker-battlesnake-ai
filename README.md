# Amazon SageMaker for Battlesnake AI

This project shows how to build and deploy an AI for the game [BattleSnake](https://play.battlesnake.com/) with [AWS Sagemaker](https://aws.amazon.com/sagemaker/)!

It is ready to deploy and contains learning materials for AI enthusiast.

__What is Battlesnake?__ (taken from [Battlesnake.com](https://docs.battlesnake.com/rules)):

> Battlesnake is an autonomous survival game where your snake competes with others to find and eat food without being eliminated. To accomplish this, you will have to teach your snake to navigate the serpentine paths created by walls, other snakes, and their own growing tail without running out of energy.

## Content

This project can be used in three steps:

- __[STEP 1 - Deploy a pretrained AI](Documentation/DeployTheAIEndpoint.md)__ : Will deploy a Snake AI in a single click! You are ready for the competition.
- __[STEP 2 - Customize the AI heuristics](Documentation/UpdateHeuristicsAndDeploy.md)__ : Customize AI behaviour, visualize your result and publish an upgraded version!
- __[STEP 3 - Train the AI model with your own settings](Documentation/TrainModelAndDeploy.md)__ : The most challenging one: train the AI again with different settings, visualize your result and publish an upgraded version!

_You have to do STEP 1 in order to be able to do STEP 2 or STEP 3._

### Architecture

If you use  STEP 1, STEP 2 and STEP 3, you will have the following deployed:

![General Architecture](Documentation/images/ArchitectureSagemakerBattleSnakeFull.png "General Architecture")

### Cost

This project have been design to fit inside the AWS free tiers for some time.
See each section for duration and cost.

## License

This project is licensed under the Apache-2.0 License.
