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

'''
def create_environment(env_config):
    # This import must happen inside the method so that worker processes import this code
    import ma_battlesnake
    return MultiAgentBattlesnake(num_agents=5, map_height=11)
'''

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
            
    def register_env_creator(self):
                register_env("MultiAgentBattlesnake-v1", lambda _: MultiAgentBattlesnake(num_agents=self.num_agents, 
                                                                                         map_height=self.map_height))
    # Callback function that is executed after each training iteration
    # Used to implement curriculum learning, inject custom metrics, etc.
    def on_train_result(self, info):
        # Add reformatted metrics for SageMaker
        info['result']['sm__episode_len_mean'] = info['result']['episode_len_mean']
        info['result']['sm__episode_reward_min'] = info['result']['episode_reward_min']
        info['result']['sm__episode_reward_max'] = info['result']['episode_reward_max']
        info['result']['sm__episode_reward_mean'] = info['result']['episode_reward_mean']
        info['result']['sm__policy_0_reward_max'] = info['result']['policy_reward_max']['policy_0']
        info['result']['sm__policy_0_reward_min'] = info['result']['policy_reward_min']['policy_0']
        info['result']['sm__policy_0_reward_mean'] = info['result']['policy_reward_mean']['policy_0']

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
        else:
            eff_map_size = self.map_height

        info['result']['sm__effective_map_size'] = eff_map_size

        trainer = info["trainer"]
        trainer.workers.foreach_worker(
                lambda ev: ev.foreach_env(
                    lambda env: env.set_effective_map_size(eff_map_size)))

    def get_experiment_config(self):        
        tmp_env = MultiAgentBattlesnake(num_agents=self.num_agents, map_height=self.map_height)
        policies = {'policy_{}'.format(i): (None, tmp_env.observation_space, tmp_env.action_space, {}) for i in range(self.num_agents)}
        policy_ids = list(policies.keys())
        
        ModelCatalog.register_custom_model("my_model", VisionNetwork)
        
        configs = {
                'callbacks': { 
                    'on_train_result': self.on_train_result,
                },
                'num_workers': (self.num_cpus-1),
                'num_envs_per_worker': 1,
                'num_gpus': self.num_gpus,
                "num_gpus_per_worker": 0,
                'model': 
                {
                    "custom_model": "my_model", 
                    'use_lstm': False, 
                    "max_seq_len": 60,
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
            "checkpoint_freq": 50,
            'config': {**configs, **self.additional_configs}
          }
        }
    
if __name__ == "__main__":
    
    MyLauncher().train_main()