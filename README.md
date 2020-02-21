# Amazon SageMaker for Battlesnake AI

This project shows how to build and deploy an AI for the game [BattleSnake](https://play.battlesnake.com/) with [AWS Sagemaker](https://aws.amazon.com/sagemaker/)!

It is ready to deploy and contains learning materials for AI enthusiast.

__What is Battlesnake__

Taken from [Battlesnake.com](https://docs.battlesnake.com/rules):

> Battlesnake is an autonomous survival game where your snake competes with others to find and eat food without being eliminated. To accomplish this, you will have to teach your snake to navigate the serpentine paths created by walls, other snakes, and their own growing tail without running out of energy.

## Content

This project can be used in three steps:

- __[Deploy a pretrained AI](Documentation/DeployTheAIEndpoint.md)__ : Will deploy a Serverless endpoint in a single click! You are ready for the competition.
- __[Train the AI model with your own settings](Documentation/TrainModelAndDeploy.md#Training-a-reinforcement-learning-model)__ : The most challenging one: train the AI again with different settings, visualize your result and publish an upgraded version!
- __[Customize the AI heuristics](Documentation/TrainModelAndDeploy.md#Heuristics-development)__ : Customize AI behaviour, visualize your result and publish an upgraded version!

_We recommend doing the steps in order but you can also jump on the one you like directly._

## Package Developers

[Here](https://github.com/awslab/sagemaker-battlesnake-ai/Documentation/PackageDeveloperDoc.md) are instructions that have been used to create the initial Lambda package.