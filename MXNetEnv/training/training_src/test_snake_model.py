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

import numpy as np
import argparse
import time
import mxnet as mx

from battlesnake_gym.snake_gym import BattlesnakeGym
from examples.networks.dqn_agent_mxnet import Agent_mxnet
from examples.networks.utils import sort_states_for_snake_id

ctx = mx.gpu(0) if mx.context.num_gpus() > 0 else mx.cpu()

def run(args):
    max_time_steps = 1000
    
    map_size = tuple(args.map_size)
    number_of_snakes = args.number_of_snakes
    state_shape = (map_size[0], map_size[1], (1+number_of_snakes))

    env = BattlesnakeGym(map_size=map_size, number_of_snakes=number_of_snakes)

    agents = []

    for i in range(number_of_snakes):
        agent = Agent_mxnet(state_shape=state_shape, action_size=4, seed=0)
        agent.qnetwork_local.load_parameters(args.model_name.format(i),
                                             ctx=ctx)
    
    state = env.reset()
    for t in range(max_time_steps):
        actions = []
        for i in range(number_of_snakes):
            state_agent_i = sort_states_for_snake_id(state, i+1)
            action = agent.act(state_agent_i, eps=0)
            actions.append(action)
        next_state, reward, done, _ = env.step(np.array(actions))
        env.render(mode=args.render_mode)
        time.sleep(0.2)
        state = next_state
        if done:
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train a DQG agent for the Battlesnake.IO')

    parser.add_argument('--map_size', nargs="+", type=int, default=[16, 16],
                        metavar="MS", help='map size')
    parser.add_argument('--number_of_snakes', type=int, default=1, metavar="NS",
                        help='Number of snakes')
    parser.add_argument('--model_name', type=str,
                        default="params/checkpoint{}.pth",
                        metavar='MN',
                        help='filepath of the model parameters')
    parser.add_argument('--render_mode', type=str, default="ascii",
                        metavar='MN',
                        help='Render mode options: ["ascii", "Human"]')

    args = parser.parse_args()
    run(args)
    
