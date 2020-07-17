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
from battlesnake_gym.rewards import SimpleRewards

try:
    from utils import sort_states_for_snake_id
except ModuleNotFoundError:
    from training.training_src.utils import sort_states_for_snake_id

try:
    from battlesnake_heuristics import MyBattlesnakeHeuristics
except ModuleNotFoundError:
    from inference.inference_src.battlesnake_heuristics import MyBattlesnakeHeuristics
    
## MultiAgentEnv wrapper for battlesnake_gym
class MultiAgentBattlesnake(MultiAgentEnv):

    MAX_MAP_HEIGHT = 21
        
    def __init__(self, num_agents, map_height, heuristics, rewards=SimpleRewards()):
        observation_type = "max-bordered-51s"
         
        self.env = BattlesnakeGym(
            observation_type=observation_type,
            number_of_snakes=num_agents, 
            map_size=(map_height, map_height), rewards=rewards)
        
        self.observation_height = self.MAX_MAP_HEIGHT
        self.action_space = self.env.action_space[0]
        
        gym_observation_space = gym.spaces.Box(low=-1.0, high=5.0,
                                               shape=(self.observation_height, self.observation_height, 6), dtype=np.float32)

        self.observation_space = gym.spaces.Dict({
            "action_mask": gym.spaces.Box(0, 1, shape=(4,),
                dtype=np.float32),
            "state": gym_observation_space})

        self.num_agents = num_agents
        self.observation_type = observation_type
        self.old_obs1 = {}
        self.heuristics = heuristics
        if len(self.heuristics) > 0:
            self.battlesnake_heuristics = MyBattlesnakeHeuristics()
            self.heuristics_list = []
            for heuristic_name in self.heuristics:
                if heuristic_name == "banned_forbidden_moves":
                    self.heuristics_list.append(self.battlesnake_heuristics.banned_forbidden_moves)
                elif heuristic_name == "banned_wall_hits":
                    self.heuristics_list.append(self.battlesnake_heuristics.banned_wall_hits)
        self.rewards = rewards
        
    def set_effective_map_size(self, eff_map_size):
        self.__init__(self.num_agents, eff_map_size, self.heuristics, self.rewards)
        self.reset()

    def reset(self):
        self.mask = {}
        new_obs, _, _, info = self.env.reset()

        obs = {}

        # add empty map placeholders for use until we've seen 2 steps
        empty_map = np.zeros((self.observation_height, self.observation_height, 3))
        
        new_obs = np.array(new_obs, dtype=np.float32)

        for i in range(self.num_agents):
            agent_id = "agent_{}".format(i)
            
            obs_i = sort_states_for_snake_id(new_obs, i+1)
            
            merged_map = np.concatenate((empty_map, obs_i), axis=-1)

            if len(self.heuristics) > 0:
                health = {k: 100 for k in range(self.num_agents)}
                mask = self.battlesnake_heuristics.get_action_masks_from_functions(
                    obs_i, i, 0, health, self.env, 
                    functions=self.heuristics_list)
            else:
                mask = np.array([1, 1, 1, 1])
            obs[agent_id] = {"state": merged_map, "action_mask": mask}
            
            self.mask[agent_id] = obs[agent_id]["action_mask"]
            self.old_obs1[agent_id] = obs_i 
            
        return obs

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
            
            obs_i = np.array(o, dtype=np.float32)
            obs_i = sort_states_for_snake_id(obs_i, i+1)
            
            merged_map = np.concatenate((old_obs1, obs_i), axis=-1)
            
            infos[key] = info
            rewards[key] = r[i]
            if len(self.heuristics) > 0 and self.env.snakes.get_snakes()[i].is_alive():
                turn_count = info["current_turn"]+1
                health = info["snake_health"]

                mask = self.battlesnake_heuristics.get_action_masks_from_functions(
                        obs_i, i, turn_count, health, self.env, 
                        functions=self.heuristics_list)
                                
            else:
                mask = np.array([1, 1, 1, 1])

            obs[key] = {"state": merged_map, "action_mask": mask}
            self.old_obs1[key] = np.array(obs_i, dtype=np.float32)
            
            self.mask[key] = obs[key]["action_mask"]

        dead_count = 0
        for x in range(self.num_agents):
            if d[x] == True:
                dead_count += 1

        dones = {'__all__': dead_count >= self.num_agents-1}

        return obs, rewards, dones, infos
