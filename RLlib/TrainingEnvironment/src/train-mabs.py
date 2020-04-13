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
        self.map_height = 7
            
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
        
        iteration = iteration % 70

        if iteration <= 10:
            eff_map_size = 7
        elif iteration <= 20:
            eff_map_size = 9
        elif iteration <= 30:
            eff_map_size = 11
        elif iteration <= 40:
            eff_map_size = 13
        elif iteration <= 50:
            eff_map_size = 15
        elif iteration <= 60:
            eff_map_size = 17
        else:
            eff_map_size = 19

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
        
        return {
          "training": { 
            "env": "MultiAgentBattlesnake-v1",
#            "restore": "/opt/ml/code/checkpoints/checkpoint_250/checkpoint-250",
            "run": "PPO",
            "stop": {
              "training_iteration": self.num_iters,
            },
            "checkpoint_freq": 50,
            'config': {
                'callbacks': { 
                    'on_train_result': self.on_train_result,
                },
                'monitor': False,  # Record videos.
                'lambda': 0.90,
                'gamma': 0.999,
                'kl_coeff': 0.2,
                'clip_rewards': True,
                'vf_clip_param': 175.0,
                'train_batch_size': 9216,
                'sample_batch_size': 96,
                'sgd_minibatch_size': 256,
                'num_sgd_iter': 3,
                'num_workers': (self.num_cpus-1),
                'num_envs_per_worker': 1,
                'batch_mode': 'complete_episodes',
                'observation_filter': 'NoFilter',
                'vf_share_layers': False,
                'num_gpus': self.num_gpus,
                "num_gpus_per_worker": 0,
                'lr': 5.0e-4,
                'log_level': 'ERROR',
                'simple_optimizer': False,
                'model': {"custom_model": "my_model", 
                    'use_lstm': False, 
                    "max_seq_len": 60,
                },
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