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

class Rewards:
    '''
    Base class to set up rewards for the battlesnake gym
    '''
    def get_reward(self, name, snake_id, episode):
        raise NotImplemented()

class SimpleRewards(Rewards):
    '''
    Simple class to handle a fixed reward scheme
    '''
    def __init__(self):
        self.reward_dict = {"another_turn": 1,
                            "ate_food": 0,
                            "won": 0,
                            "died": 0,
                            "ate_another_snake": 0,
                            "hit_wall": 0,
                            "hit_other_snake": 0,
                            "hit_self": 0,
                            "was_eaten": 0,
                            "other_snake_hit_body": 0,
                            "forbidden_move": 0,
                            "starved": 0}

    def get_reward(self, name, snake_id, episode):
        return self.reward_dict[name]
