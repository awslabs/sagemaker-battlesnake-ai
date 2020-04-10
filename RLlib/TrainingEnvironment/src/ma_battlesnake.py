from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os

import gym
import ray
import tensorflow as tf

from ray import tune
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog

from battlesnake_gym.snake_gym import BattlesnakeGym
from utils import sort_states_for_snake_id

## MultiAgentEnv wrapper for battlesnake_gym
class MultiAgentBattlesnake(MultiAgentEnv):

    MAX_MAP_HEIGHT = 21
    def __init__(self, num_agents, map_height):
        observation_type = "max-bordered-51s"
        self.env = BattlesnakeGym(
            observation_type=observation_type,
            number_of_snakes=num_agents, 
            map_size=(map_height, map_height))
        
        if "bordered" in observation_type:
            if "max-bordered" in observation_type:
                self.observation_height = self.MAX_MAP_HEIGHT
            else: # If only bordered with 2 rows of -1
                self.observation_height = map_height + 2
        else: # Flat without border
            self.observation_height = map_height

        self.action_space = self.env.action_space[0]

        self.observation_space = gym.spaces.Box(low=-1.0, high=5.0, shape=(self.observation_height, self.observation_height, 6), dtype=np.float32)
        self.num_agents = num_agents
        self.observation_type = observation_type
        self.old_obs1 = {}

    def set_effective_map_size(self, eff_map_size):
        self.__init__(self.num_agents, eff_map_size)
        self.reset()

    def reset(self):
        new_obs, _, _, info = self.env.reset()

        obs = {}

        # add empty map placeholders for use until we've seen 2 steps
        empty_map = np.zeros((self.observation_height, self.observation_height, 3))
        
        new_obs = np.array(new_obs, dtype=np.float32)

        for i in range(self.num_agents):
            agent_id = "agent_{}".format(i)
            
            obs_i = sort_states_for_snake_id(new_obs, i+1)
            
            merged_map = np.concatenate((empty_map, obs_i), axis=-1)

            if self.num_agents > 1:
                obs[agent_id] = merged_map
            else:
                obs[agent_id] = merged_map
                
            self.old_obs1[agent_id] = obs_i 

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

        for i, key in enumerate(sorted(action_dict.keys())):            
            old_obs1 = self.old_obs1[key]
            
            o_i = np.array(o, dtype=np.float32)
            o_i = sort_states_for_snake_id(o_i, i+1)
            
            merged_map = np.concatenate((old_obs1, o_i), axis=-1)
            
            infos[key] = info
            if self.num_agents > 1:
                rewards[key] = r[i]
                obs[key] = merged_map

            else:
                rewards[key] = r
                obs[key] = merged_map

            self.old_obs1[key] = np.array(o_i, dtype=np.float32)

        dead_count = 0
        for x in range(self.num_agents):
            if d[x] == True:
                dead_count += 1

        dones = {'__all__': dead_count >= self.num_agents-1}

        return obs, rewards, dones, infos
