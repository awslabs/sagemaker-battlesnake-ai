import json
import os

import gym
import ray
import ray.tune
import json
from ray.tune import run_experiments
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from cnn_tf import VisionNetwork
from ma_battlesnake import MultiAgentBattlesnake

from sagemaker_rl.ray_launcher import SageMakerRayLauncher
from battlesnake_gym.rewards import SimpleRewards

class MyLauncher(SageMakerRayLauncher):
    def __init__(self):
        super().__init__()
        self.hparams = json.loads(os.environ.get("SM_HPS", "{}"))
        self.num_agents = self.hparams['num_agents']
        self.num_iters = self.hparams['num_iters']
        self.iterate_map_size = self.hparams['iterate_map_size']
        self.map_height = self.hparams['map_size']
        self.algorithm = self.hparams['algorithm']
        self.additional_configs = self.hparams["additional_configs"]
        self.converter = {"Snake hit wall": "Snake_hit_wall",
                          "Snake was eaten - same tile": "Snake_was_eaten",
                          "Snake was eaten - adjacent tile": "Snake_was_eaten",
                          "Other snake hit body": "Killed_another_snake",
                          "Snake hit body - hit itself": "Snake_hit_body",
                          "Snake hit body - hit other": "Snake_hit_body",
                          "Did not collide": "",
                          "Ate another snake": "Killed_another_snake",
                          "Dead": "",
                          "Starved": "Starved",
                          "Forbidden move": "Forbidden_move"
        }
        self.rewards = SimpleRewards()
        if "rewards" in self.hparams:
            self.rewards.reward_dict = self.hparams["rewards"]
            
        self.heuristics = []
        if "heuristics" in self.hparams:
            self.heuristics = self.hparams["heuristics"]
          
    def register_env_creator(self):
        register_env("MultiAgentBattlesnake-v1", lambda _: MultiAgentBattlesnake(
            num_agents=self.num_agents, 
            map_height=self.map_height,
            heuristics=self.heuristics, 
            rewards=self.rewards))

    def on_episode_start(self, info):
        for outcome in ["Snake_hit_wall", "Snake_was_eaten", "Snake_hit_body", "Killed_another_snake",
                        "Starved", "Forbidden_move"]:
            info['episode'].custom_metrics[outcome] = 0

    def on_episode_step(self, info):
        agent_info = info['episode'].last_info_for('agent_1')
        if "snake_info" in agent_info:
            for i in range(self.num_agents):
                snake_info_i = agent_info["snake_info"][i]
                converted_outcome = self.converter[snake_info_i]
                if len(converted_outcome) > 0:
                    info['episode'].custom_metrics[converted_outcome] += 1

    def on_episode_end(self, info):
        agent_info = info['episode'].last_info_for('agent_1')
        for i in range(self.num_agents):
            snake_info_i = agent_info["snake_info"][i]
            converted_outcome = self.converter[snake_info_i]
            if len(converted_outcome) > 0:
                info['episode'].custom_metrics[converted_outcome] += 1

        for i in range(self.num_agents):
            snake_max_len = info['episode'].last_info_for('agent_1')['snake_max_len']
            info['episode'].custom_metrics['policy{}_max_len'.format(i)] = snake_max_len[i]
    
    def on_train_result(self, info):
        max_lens_per_policy = []
        for i in range(self.num_agents):
            agent_id = "policy{}".format(i)
            max_lens_per_policy.append(info['result']['custom_metrics']['{}_max_len_max'.format(agent_id)])
        info['result']['best_snake_episode_len_max'] = max(max_lens_per_policy)
        info['result']['worst_snake_episode_len_max'] = min(max_lens_per_policy)

        info['result']['episode_len_max'] = max(info["result"]["hist_stats"]["episode_lengths"])
        info['result']['episode_len_min'] = min(info["result"]["hist_stats"]["episode_lengths"])

        # curriculum learning -
        # here we adjust effective map size based on current training iteration
        # you could also adjust based on mean rewards, mean episode length, etc.
        iteration = info['result']['training_iteration']
        
        if self.iterate_map_size:
            iteration = iteration % 30
            if iteration <= 10:
                eff_map_size = 7
            elif iteration <= 20:
                eff_map_size = 11
            else:
                eff_map_size = 19
            info['result']['sm__effective_map_size'] = eff_map_size

            trainer = info["trainer"]
            trainer.workers.foreach_worker(
                    lambda ev: ev.foreach_env(
                        lambda env: env.set_effective_map_size(eff_map_size)))
        else:
            eff_map_size = self.map_height

    def get_experiment_config(self):        
        tmp_env = MultiAgentBattlesnake(num_agents=self.num_agents, map_height=self.map_height, heuristics=self.heuristics)
        policies = {'policy_{}'.format(i): (None, tmp_env.observation_space, tmp_env.action_space, {}) for i in range(self.num_agents)}
        policy_ids = list(policies.keys())
        
        ModelCatalog.register_custom_model("my_model", VisionNetwork)
        
        configs = {
                'callbacks': { 
                    'on_episode_start': self.on_episode_start,
                    'on_episode_step': self.on_episode_step,
                    'on_episode_end': self.on_episode_end,
                    'on_train_result': self.on_train_result,
                },
                'num_workers': (self.num_cpus-1),
                'num_envs_per_worker': 1,
                'num_gpus': self.num_gpus,
                "num_gpus_per_worker": 0,
                'model': 
                {
                    "no_final_linear": False,
                    "vf_share_layers": False,
                    "custom_model": "my_model", 
                    'use_lstm': False, 
                    "max_seq_len": 60,
                    "conv_filters": [[16, [5, 5], 4], [32, [3, 3], 1], [256, [3, 3], 1]],
                    "custom_options": {
                        "max_map_height": 21
                    }
                },
                'multiagent': {
                    'policies': policies,
                    'policy_mapping_fn': lambda agent_id: policy_ids[int(agent_id[6:])],
                },
                'use_pytorch': False,
            }
        
        return {
          "training": { 
            "env": "MultiAgentBattlesnake-v1",
            "run": self.algorithm,
            "stop": {
              "training_iteration": self.num_iters,
            },
            "checkpoint_freq": 500,
            'config': {**configs, **self.additional_configs}
          }
        }
    
if __name__ == "__main__":
    
    MyLauncher().train_main()