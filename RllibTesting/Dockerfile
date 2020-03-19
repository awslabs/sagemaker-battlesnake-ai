FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:2.1.0-gpu-py36-cu101-ubuntu18.04

MAINTAINER Scott Perry <perrysc@amazon.com>

RUN apt-get update && apt-get install -y rsync

RUN pip install ray[rllib]

WORKDIR /

WORKDIR /opt/ml/code

COPY . . 

ENV SAGEMAKER_PROGRAM train_cnn_ppo_tf.py
