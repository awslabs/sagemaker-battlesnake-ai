from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import gym
import ray
import tensorflow as tf
from ray import tune
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env
from battlesnake_gym.snake_gym import BattlesnakeGym

#import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
#from fcnet_pytorch import FullyConnectedNetwork
#from fcnet_tf import FullyConnectedNetwork
#from cnn_pytorch import VisionNetwork
from cnn_tf import VisionNetwork
import numpy as np
import os

parser = argparse.ArgumentParser()
# Number of snakes/policies within the gym
# We're forcing 1 policy per agent here, but this doesn't have to be the case
parser.add_argument('--num-agents', type=int, default=4)
# Map size will be this, squared
parser.add_argument('--map-dim', type=int, default=11)
# Number of training iterations
parser.add_argument('--num-iters', type=int, default=10)
# Number of Ray workers. Set to #CPUs - 1 to avoid errors.
parser.add_argument('--num-workers', type=int, default=os.cpu_count()-1)
# Number of GPUs to use for learner (not rollout workers). Defaults to all GPUs.
parser.add_argument('--num-gpus', type=int, 
        default=len(tf.config.experimental.list_physical_devices('GPU')))
# Model output directory
parser.add_argument('--log-dir', default="/opt/ml/model/")
# Number of envs per worker
parser.add_argument('--num-envs', type=int, default=2)
# Learning rate
parser.add_argument('--lr', type=float, default=5.0e-4)
# Number of SGD iterations
parser.add_argument('--sgd-iters', type=int, default=3)
# PPO Lambda
parser.add_argument('--lmb', type=float, default=0.90)
# PPO Gamma
parser.add_argument('--gamma', type=float, default=0.99)
# Sample batch size
parser.add_argument('--sample-bs', type=int, default=96)
# SGD minibatch size
parser.add_argument('--sgd-mbs', type=int, default=64)
# Train batch size
parser.add_argument('--train-bs', type=int, default=11520)
# VF clip
parser.add_argument('--vf-clip', type=float, default=175.0)
# PPO KL coefficient
parser.add_argument('--kl-coeff', type=float, default=0.2)
# Checkpoint frequency (in iterations)
parser.add_argument('--cpt-freq', type=int, default=10)
# LSTM max sequence length
parser.add_argument('--lstm-max', type=int, default=60)


## MultiAgentEnv wrapper for battlesnake_gym
class RllibBattlesnake(MultiAgentEnv):

    def __init__(self, num_agents, map_height):
        self.env = BattlesnakeGym(
            observation_type="flat-51s",
            number_of_snakes=num_agents, 
            map_size=(map_height, map_height))

        self.action_space = self.env.action_space[0]

        self.observation_space = gym.spaces.Box(low=-1.0, high=5.0, shape=(map_height, map_height, (num_agents+1)*2), dtype=np.float32)
        self.num_agents = num_agents
        self.map_height = map_height

    def reset(self):
        new_obs, _, _, info = self.env.reset()

        obs = {}

        # add empty map placeholders for use until we've seen 2 steps
        empty_map = np.zeros((self.map_height, self.map_height, self.num_agents+1))
        self.old_obs1 = np.array(new_obs, dtype=np.float32)

        merged_map = np.concatenate((empty_map, self.old_obs1), axis=-1)

        for x in range(self.num_agents):
            if self.num_agents > 1:
                obs['agent_%d' % x] = merged_map
            else:
                obs['agent_%d' % x] = merged_map
        return obs

    def render(self):
        self.env.render()

    def step(self, action_dict):
        actions = []

        for key, value in sorted(action_dict.items()):
            actions.append(value)

        o, r, d, info = self.env.step(actions)
        rewards = {}
        obs = {}
        infos = {}

        merged_map = np.concatenate((self.old_obs1, np.array(o, dtype=np.float32)), axis=-1)

        for pos, key in enumerate(sorted(action_dict.keys())):
            infos[key] = info
            if self.num_agents > 1:
                rewards[key] = r[pos]
                obs[key] = merged_map

            else:
                rewards[key] = r
                obs[key] = merged_map

        self.old_obs1 = np.array(o, dtype=np.float32)

        dead_count = 0
        for x in range(self.num_agents):
            if d[x] == True:
                dead_count += 1

        dones = {'__all__': dead_count >= self.num_agents-1}

        return obs, rewards, dones, infos


if __name__ == '__main__':
    args = parser.parse_args()
    # webui_host set below to correct for a Ray issue when running in a container
    ray.init(webui_host="127.0.0.1")

    register_env('battlesnake', lambda _: RllibBattlesnake(
        args.num_agents, args.map_dim))
    single_env = RllibBattlesnake(args.num_agents, args.map_dim)
    obs_space = single_env.observation_space
    act_space = single_env.action_space

    def gen_policy(_):
        return (None, obs_space, act_space, {})

    policies = {'policy_{}'.format(i): gen_policy(i)
                for i in range(args.num_agents)}

    policy_ids = list(policies.keys())

    ModelCatalog.register_custom_model("my_model", VisionNetwork)

    tune.run(
        'PPO',
        stop={'training_iteration': args.num_iters},
        checkpoint_freq=args.cpt_freq,
        local_dir=args.log_dir,
        config={
            'env': 'battlesnake',
            'lambda': args.lmb,
            'gamma': args.gamma,
            'kl_coeff': args.kl_coeff,
            'clip_rewards': True,
            'vf_clip_param': args.vf_clip,
            'train_batch_size': args.train_bs,
            'sample_batch_size': args.sample_bs,
            'sgd_minibatch_size': args.sgd_mbs,
            'num_sgd_iter': args.sgd_iters,
            'num_workers': args.num_workers,
            'num_envs_per_worker': args.num_envs,
            'batch_mode': 'truncate_episodes',
            'observation_filter': 'NoFilter',
            'vf_share_layers': False,
            'num_gpus': args.num_gpus,
            "num_gpus_per_worker": 0,
            'lr': args.lr,
            'log_level': 'ERROR',
            'simple_optimizer': False,
            'model': {"custom_model": "my_model", 'use_lstm': True, 
                "max_seq_len": args.lstm_max },
            'multiagent': {
                'policies': policies,
                'policy_mapping_fn': tune.function(
                    lambda agent_id: policy_ids[int(agent_id[6:])]),
            },
            'use_pytorch': False,
        },
    )
