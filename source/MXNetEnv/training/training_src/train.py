# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# or in the "license" file accompanying this file. This file is distributed 
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either 
# express or implied. See the License for the specific language governing 
# permissions and limitations under the License.

import argparse
import os
import json

from battlesnake_gym.snake_gym import BattlesnakeGym
from mxboard import SummaryWriter

from dqn_run import trainer
from networks.agent import MultiAgentsCollection

def run(seed, args):
    print("Running with seed = {}".format(seed))
    map_size = json.loads(args.map_size)

    # Initialise logging
    if args.model_dir is None:
        # Check if the model is running in Sagemaker
        if "SM_MODEL_DIR" in os.environ:
            model_dir = os.environ['SM_MODEL_DIR']
        else:
            model_dir = "params"

    # Check if the model is running in Sagemaker
    load = args.load
    if 'SM_CHANNEL_WEIGHTS' in os.environ and load is not None:
        load = os.environ['SM_CHANNEL_WEIGHTS'] + "//" + load
    
    if args.writer:
        writer = SummaryWriter("logs/{}-seed{}".format(args.run_name, seed), verbose=False)
    else:
        writer = None

    # Initialise the environment
    env = BattlesnakeGym(map_size=map_size, observation_type=args.snake_representation)
    env.seed(seed)

    # Initialise agent
    if args.state_type == "layered":
        state_depth = 1+args.number_of_snakes
    elif args.state_type == "one_versus_all":
        state_depth = 3
        
    if "bordered" in args.snake_representation:
        state_shape = (map_size[0]+2, map_size[1]+2, state_depth)
    else:
        state_shape = (map_size[0], map_size[1], state_depth)

    agent_params = (seed, model_dir,
                    load, args.load_only_conv_layers,
                    args.models_to_save,
                    # State configurations
                    args.state_type, state_shape, args.number_of_snakes,

                    # Learning configurations
                    args.buffer_size, args.update_every,
                    args.lr_start, args.lr_step, args.lr_factor,
                    args.gamma, args.tau, args.batch_size, 

                    # Network configurations
                    args.qnetwork_type, args.sequence_length,
                    args.starting_channels, args.number_of_conv_layers,
                    args.number_of_dense_layers, args.number_of_hidden_states,
                    args.depthS, args.depth,
                    args.kernel_size, args.repeat_size,
                    args.activation_type)
    
    agent = MultiAgentsCollection(*agent_params)

    trainer(env, agent, args.number_of_snakes,
            args.run_name, args.episodes,
            args.max_t, args.warmup, 
            args.eps_start, args.eps_end, args.eps_decay,
        
            args.print_score_steps,
            args.save_only_best_models,
            args.save_model_every,
            args.render_steps, args.should_render,
            writer, args.print_progress)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train a DQN agent for the Battlesnake.IO')

    parser.add_argument('--seeds', nargs="+", type=int, default=[0, 666, 15],
                        help='seed for randomiser. Code will run n times given the number of seeds')

    # Gym configurations
    parser.add_argument('--map_size', type=str, default="[15, 15]",
                        help='Size of the battlesnake map, default (15, 15)')
    parser.add_argument('--number_of_snakes', type=int, default=4, help='Number of snakes')

    # Representation configurations
    parser.add_argument('--snake_representation', type=str, default="bordered-51s",
                        help="how to represent the snakes and the gym, default bordered-51s, options: [\"flat-num\", \"bordered-num\", \"flat-51s\", \"bordered-51s\"])")
    parser.add_argument('--state_type', type=str, default="one_versus_all",
                        help='Output option of the state, default: layered, options: ["layered", "one_versus_all"]')
    
    # Training configurations
    parser.add_argument('--buffer_size', type=int, default=5000,
                        help='Buffer size (default: 5000)')
    parser.add_argument('--update_every', type=int, default=20,
                        help='Episodes to update network (default 20)')
    parser.add_argument('--lr_start', type=float, default=0.0005,
                        help='Starting learning rate (default: 0.0005)')
    parser.add_argument('--lr_step', type=int, default=5e5,
                        help='Number of steps for learning rate decay (default: 50k)')
    parser.add_argument('--lr_factor', type=float, default=0.5,
                        help='Factor to decay learning rate (default: 0.5)')
    parser.add_argument('--gamma', type=float, default=0.95,
                        help='discount factor (default: 0.95)')
    parser.add_argument('--tau', type=float, default=1e-3,
                        help='soft update factor (default: 1e-3)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size (default: 64)')
    parser.add_argument('--episodes', type=int, default=100000,
                        help='Number of espidoes (default: 100000)')
    parser.add_argument('--max_t', type=int, default=1000,
                        help='Max t (default: 1k)')
    parser.add_argument('--eps_start', type=float, default=1.0,
                        help='Episilon start (default: 1.0)')
    parser.add_argument('--eps_end', type=float, default=0.01,
                        help='Episilon end (default: 0.01)')
    parser.add_argument('--eps_decay', type=float, default=0.995,
                        help='Episilon decay (default: 0.995)')
    parser.add_argument('--warmup', type=float, default=0,
                        help='Warmup (default: 0)')

    # Network configurations
    parser.add_argument('--load', default=None, help='Load from param file')
    parser.add_argument('--load_only_conv_layers', default=False,
                        help='Boolean to define if only the convolutional layers should be loaded')
    parser.add_argument('--qnetwork_type', default="attention",
                        help='Type of q_network. Options: ["concat", "attention"]')
    parser.add_argument('--starting_channels', type=int, default=6,
                        help='starting channels for qnetwork')
    parser.add_argument('--number_of_conv_layers', type=int, default=3,
                        help='Number of conv. layers for qnetwork concat')
    parser.add_argument('--number_of_dense_layers', type=int, default=2,
                        help='Number of dense layers for qnetwork concat')
    parser.add_argument('--depthS', type=int, default=10,
                        help='depth of the embeddings for the snake ID for qnetwork attention')
    parser.add_argument('--depth', type=int, default=200,
                        help='depth of the embeddings for the snake health for qnetwork attention')
    parser.add_argument('--number_of_hidden_states', type=int, default=128,
                        help='Number of hidden states in the qnetwork')
    parser.add_argument('--kernel_size', type=int, default=3,
                        help='kernel size for the qnetwork')
    parser.add_argument('--repeat_size', type=int, default=3,
                        help='Size to repeat input states')
    parser.add_argument('--activation_type', type=str, default="softrelu",
                        help='Activation for qnetwork')
    parser.add_argument('--sequence_length', type=int, default=2,
                        help='Number of states to feed sequencially feed in')

    # Logging information
    parser.add_argument('--print_score_steps', type=int, default=100,
                        help='Steps to print score (default: 100)')
    parser.add_argument('--models_to_save', type=str, default='all',
                       help='select which models to save options ["all", "local"] (default: all)')
    parser.add_argument('--save_only_best_models', type=bool, default=False,
                        help='Save only the best models')
    parser.add_argument('--save_model_every', type=int, default=2000,
                        help='Steps to save the model')
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--render_steps', type=int, default=1000, 
                        help='Steps to render (default: 1000)')
    parser.add_argument('--should_render', action='store_true',
                        help='render the environment to generate a gif in /gifs')
    parser.add_argument('--writer', action='store_true',
                        help='should write to tensorboard')
    parser.add_argument('--print_progress', action='store_true',
                        help='should print every progressive step')
    parser.add_argument('--run_name', type=str, default="run", 
                        help='Run name to save reward (default: run+seed)')
    
    args = parser.parse_args()
    seeds = list(args.seeds)

    for seed in seeds:
        run(seed, args)
