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
        self.map_height = self.hparams['map_height']
        self.num_iters = self.hparams['num_iters']
    
    
    def register_env_creator(self):
                register_env("MultiAgentBattlesnake-v1", lambda _: MultiAgentBattlesnake(num_agents=self.num_agents, 
                                                                                         map_height=self.map_height))
        
    def get_experiment_config(self):        
        tmp_env = MultiAgentBattlesnake(num_agents=self.num_agents, map_height=self.map_height)
        policies = {'policy_{}'.format(i): (None, tmp_env.observation_space, tmp_env.action_space, {}) for i in range(self.num_agents)}
        policy_ids = list(policies.keys())
        
        ModelCatalog.register_custom_model("my_model", VisionNetwork)
        
        return {
          "training": { 
            "env": "MultiAgentBattlesnake-v1",
            "run": "PPO",
            "stop": {
              "training_iteration": self.num_iters,
            },
            'config': {
                'monitor': False,  # Record videos.
                'lambda': 0.90,
                'gamma': 0.999,
                'kl_coeff': 0.2,
                'clip_rewards': True,
                'vf_clip_param': 175.0,
                'train_batch_size': 11520,
                'sample_batch_size': 96,
                'sgd_minibatch_size': 64,
                'num_sgd_iter': 3,
                'num_workers': (self.num_cpus-1),
                'num_envs_per_worker': 2,
                'batch_mode': 'truncate_episodes',
                'observation_filter': 'NoFilter',
                'vf_share_layers': False,
                'num_gpus': self.num_gpus,
                "num_gpus_per_worker": 0,
                'lr': 5.0e-4,
                'log_level': 'ERROR',
                'simple_optimizer': False,
                'model': {"custom_model": "my_model", 'use_lstm': True, "max_seq_len": 60 },
                'multiagent': {
                    'policies': policies,
                    'policy_mapping_fn': lambda agent_id: policy_ids[int(agent_id[6:])],
                },
                'use_pytorch': False,
            }
          }
        }

    
if __name__ == "__main__":
    
    MyLauncher().train_main()
