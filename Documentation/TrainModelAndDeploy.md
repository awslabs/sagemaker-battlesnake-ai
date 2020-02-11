# battlesnake_gym
Taken from [Battlesnake.com](https://docs.battlesnake.com/rules):

> Battlesnake is an autonomous survival game where your snake competes with others to find and eat food without being eliminated. To accomplish this, you will have to teach your snake to navigate the serpentine paths created by walls, other snakes, and their own growing tail without running out of energy.

This is an openAI gym to train your battlesnake bots (https://play.battlesnake.com/). 

## Dependencies
- gym: `pip install gym`

- array2gif: `pip install array2gif`

- mxnet (to run examples): https://mxnet.apache.org/get_started

## Installation the gym
- `pip install -e .`
TODO: include pip install for the gym

## Training the MXNet example on sagemaker
- The main entry point (sagemaker endpoint) of the training the model is [`examples/train.py`](https://github.com/jonomon/battlesnake_gym/blob/master/examples/train.py)
- The script could be ran with `python examples/train.py --should_render --print_progress --number_of_snakes 4`
- To train on Sagemaker, firstly clone this repository in a sagemaker terminal and run `build.sh` in a sagemaker terminal
- Open and run the notebook in `Sagemaker.ipynb`, endpoints of the models will be automatically created.

## Information
The gym was designed to follow the rules as provided here: https://docs.battlesnake.com/rules.

### Observation space: 
This gym provide several options for the options for the observation space. 
The observation space provided by the gym is of size `N x M x C` where `N` and `M` are the width and height of the map and `C` is the number of snakes + 1 to account for the food). The food is indicated by values of `1` in `C=0`. The snakes in `C=1 to C=num_snakes+1` and represented by 2 [options](https://github.com/jonomon/battlesnake_gym/blob/master/battlesnake_gym/snake_gym.py) (`51s`, `num`): 

*Figure 1: 51s snake representation*             |  *Figure 2: num snake representation*
:-----------------------------------------------:|:----------------------------------------------------------------:
![alt text](https://github.com/jonomon/battlesnake_gym/blob/master/images/51s.png "51s snake representation") |  ![alt text](https://github.com/jonomon/battlesnake_gym/blob/master/images/num.png "num snake representation")

The gym also provides an option to increase the map size by 2 to include -1 in the border.

*Figure 3: bordered 51s snake representation*

![alt text](https://github.com/jonomon/battlesnake_gym/blob/master/images/border.png "Bordered 51s snake representation")

### Actions:
For each snake, the possible [actions](https://github.com/jonomon/battlesnake_gym/blob/master/battlesnake_gym/snake.py) are UP, DOWN, LEFT, RIGHT (0, 1, 2, 3 respectively). Please note that according to the rules of Battsnake, if your snake is facing UP and your snake performs a DOWN action, your snake will die.

### Food spawning:
The food spawning were not provided in the official battlesnake rules. The gym was designed based on the code provided [here](
https://github.com/battlesnakeio/engine/blob/master/rules/tick.go#L82)
  
## Interacting with the gym
Based on the openAI gym framework, the following functions are used to interact with the gym:

1. >`state, _, dones, info = env.reset()`

info provides information on the turn count and the health of each snake

2. > `env.step(actions)`

`actions` expects an numpy array of size `num_snakes` containing integers 0 to 3.

3. > env.render(mode="rgb_array")

`mode` can be `rbg_array`, `ascii`, `human` 
- `rbg_array` outputs an expanded numpy array that can be used for creating gifs
- `ascii` outputs a text based representation that can be printed in the command prompt
- `human` an openAI plot will be generated

## The DQN Network
### Running DQN example
`python examples/train.py --should_render --print_progress --number_of_snakes 4`
*please refer to https://github.com/jonomon/battlesnake_gym/blob/master/examples/train.py for the other hyperparameters*

This code uses multi-agent DQN to train the bots. The `N` snakes shared the same qnetwork and the network was configured as follows:

*Figure 4: The qnetwork*
![alt text](https://github.com/jonomon/battlesnake_gym/blob/master/images/qnetwork.png "qnetwork")

Given a state representation `N x M x C` to get the action of snake `i`, the network expects that `C=1` is the representation of snake `i` and `C=2 ... num_snakes+1` are the remaining snakes (note that `C=0` represents the food).

The inputs snake health and snake ID in the Figure 4 correspond to that of Snake `i`.

## Rewards
Designing the reward function could an avenue of exploration. Currently the gym records the following events: 
1. Surviving another turn (labelled as: `"another_turn"`)
3. Eating food (labelled as: `"ate_food"`)
4. Winning the game (labelled as: `"won"`)
5. Losing the game (labelled as: `"died"`)
6. Eating another snake (labelled as: `"ate_another_snake"`)
7. Dying by hitting a wall (labelled as: `"hit_wall"`)
8. Hitting another snake (labelled as: `"hit_other_snake"`)
9. Hitting yourself (labelled as: `"hit_self"`)
10. Was eaten by another snake (labelled as: `"was_eaten"`)
11. Another snake hits your body (labelled as: `"other_snake_hit_body"`)
12. Performing a forbidden move (i.e., moving south when facing north) (labelled as: `"forbidden_move"`)
13. Dying by starving (labelled as: `"starved"`)

The current reward function is simple (`"another_turn"=1, "won"=2, "died"=-3, "forbidden_move"=-1`).
More complex reward functions with methods that could handle sparse rewards may be greatly beneficial. Also, it is possible to design different rewards for different snakes ([example](https://github.com/jonomon/battlesnake_gym/blob/save_only_best/battlesnake_gym/rewards.py)). 
 
## Testing the environment

Render env:
`BATTLESNAKE_RENDER=1 python -m unittest test.test_battlesnake_gym`

Without env rendering:
`BATTLESNAKE_RENDER=0 python -m unittest test.test_battlesnake_gym`

Performance tests:
`python -m test.measure_performance`
