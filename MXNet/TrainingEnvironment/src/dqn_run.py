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

import os
import random
import mxnet as mx
import numpy as np
from collections import deque
import json
from array2gif import write_gif

def trainer(env, agents, number_of_snakes, name,
            n_episodes, max_t, warmup,
            eps_start, eps_end, eps_decay,
            print_score_steps,
            save_only_best_models,
            save_model_every,
            render_steps,
            should_render, writer, print_progress):
    """Deep Q-Learning.

    Inspired from torch code provided in 
    https://github.com/udacity/deep-reinforcement-learning/blob/master/dqn/solution/Deep_Q_Network_Solution.ipynb
    """
    scores = [[] for _ in range(number_of_snakes)]
    scores_window = [deque(maxlen=100) for _ in range(number_of_snakes)]
    timesteps = deque(maxlen=100)
    
    eps = eps_start
    max_time_steps = 0
    
    for i_episode in range(1, n_episodes+1):
        state, _, dones, info = env.reset()
        info["episodes"] = i_episode
        score = [0 for _ in range(number_of_snakes)]
        rgb_arrays = []
        agents.reset()
        for t in range(max_t):
            
            actions = agents.get_actions(state, dones, info, t, eps)
            next_state, reward, dones, info = env.step(actions)
            info["episodes"] = i_episode

            if i_episode > warmup:
                should_learn = True
            else:
                should_learn = False

            agents.step(state, actions, reward, next_state, dones, info, t,
                        should_learn)
            for i in range(number_of_snakes):
                score[i] += reward[i]
                
            state = next_state
            if should_render and (i_episode % render_steps == 0):
                rgb_array = env.render(mode="rgb_array")
                rgb_arrays.append(rgb_array)

            number_of_snakes_alive = sum(list(dones.values()))
            if number_of_snakes - number_of_snakes_alive <= 1:
                break
            
        if should_render and (i_episode % render_steps == 0):
            write_gif(rgb_arrays, 'gifs/gif:{}-{}.gif'.format(name, i_episode),
                      fps=5)

        timesteps.append(env.turn_count)
        for i in range(number_of_snakes):
            scores_window[i].append(score[i])
            scores[i].append(score[i])
            
        eps = max(eps_end, eps_decay*eps)
        if writer:
            for i in range(number_of_snakes):
                writer.add_scalar("rewards_{}".format(i), score[i], i_episode)
            writer.add_scalar("max_timesteps", t, i_episode)

        average_score = ""
        for i in range(number_of_snakes):
            score_window = scores_window[i]
            average_score += "\t{:.2f}".format(np.mean(score_window))

        print_string = 'Episode {}\tAverage Score: {}\tMean timesteps {:.2f}'.format(
                i_episode, average_score, np.mean(timesteps))
        if print_progress:
            print("\r"+print_string, end="")
        if i_episode % print_score_steps == 0:
            if print_progress:
                print("\r"+print_string)
            else:
                print(print_string)

        if save_only_best_models:
            if env.turn_count > max_time_steps:
                if i_episode % save_model_every == 0 and i_episode > 200:
                    max_time_steps = env.turn_count
                    agents.save(name, i_episode)
        else:
            if i_episode % save_model_every == 0:
                agents.save(name, i_episode)
