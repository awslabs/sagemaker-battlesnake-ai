from collections import deque, namedtuple
import random
import multiprocessing
from multiprocessing.pool import ThreadPool as Pool

import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon

from networks.qnetworks import QNetworkConcat, QNetworkAttention, QNetworkVision
from networks.utils import sort_states_for_snake_id

ctx = mx.gpu(0) if mx.context.num_gpus() > 0 else mx.cpu()

class MultiAgentsCollection:
    def __init__(self, seed, model_dir,
                 load, load_only_conv_layers,
                 models_to_save,
                 # State configurations
                 state_type, state_shape, number_of_snakes,

                 # Learning configurations
                 buffer_size, update_every,
                 lr_start, lr_step, lr_factor,
                 gamma, tau, batch_size, 

                 # Network configurations
                 qnetwork_type, sequence_length,
                 starting_channels, number_of_conv_layers,
                 number_of_dense_layers,
                 dS, d,
                 number_of_hidden_states,
                 kernel_size, repeat_size,
                 activation_type):
        self.one_versus_all = state_type == "one_versus_all"
        self.models_to_save = models_to_save
            
        action_size = 4
        network_params = (seed, load, load_only_conv_layers,
                          qnetwork_type, state_shape, action_size,
                          sequence_length,
                          starting_channels, number_of_conv_layers,
                          number_of_dense_layers, dS, d,
                          number_of_hidden_states,
                          kernel_size, repeat_size,
                          activation_type)
        qnetwork_local = self.get_q_network("local", *network_params)
        qnetwork_target = self.get_q_network("target", *network_params)
        
        global agents
        agents = {}
        for agent_id in range(number_of_snakes):
            agent = Agent(agent_id, state_shape, action_size, seed,
                          sequence_length, buffer_size, update_every,
                          lr_start, lr_step, lr_factor,
                          gamma, tau, batch_size,
                          qnetwork_local, qnetwork_target)
            agents[agent_id] = agent
        self.model_dir = model_dir

    def get_q_network(self, network_type, seed, 
                      load, load_only_conv_layers,
                      qnetwork_type, state_shape, action_size,
                      sequence_length, starting_channels, number_of_conv_layers,
                      number_of_dense_layers, dS, d, number_of_hidden_states,
                      kernel_size, repeat_size, activation_type):
        if qnetwork_type == "attention":
            qnetwork_params = [state_shape, action_size, starting_channels,
                               dS, d, number_of_hidden_states, kernel_size,
                               repeat_size, activation_type, sequence_length,
                               seed]
        elif qnetwork_type == "concat":
            qnetwork_params = [state_shape, action_size, starting_channels,
                               number_of_conv_layers, number_of_dense_layers,
                               number_of_hidden_states, kernel_size,
                               repeat_size, activation_type, sequence_length,
                               seed]
        elif qnetwork_type == "vision":
            qnetwork_params = [state_shape, action_size, starting_channels,
                               number_of_conv_layers, number_of_dense_layers,
                               number_of_hidden_states, kernel_size,
                               repeat_size, activation_type, sequence_length,
                               seed]

        if qnetwork_type == "concat":
            qnetwork = QNetworkConcat(*qnetwork_params)
        elif qnetwork_type == "attention":
            qnetwork = QNetworkAttention(*qnetwork_params)
        elif qnetwork_type == "vision":
            qnetwork = QNetworkVision(*qnetwork_params)

        qnetwork.hybridize(static_alloc=True, static_shape=True)

        if load is not None:
            if load_only_conv_layers:
                qnetwork.load_only_conv_layers(load.format(network_type))
            else:
                qnetwork.load_parameters(load.format(network_type), ctx=ctx)
        return qnetwork

    def get_actions(self, state, dones, info, turn_count, eps):
        actions = {}

        for i, agent in agents.items():
            if dones[i]:
                continue
            state_agent_i = sort_states_for_snake_id(
                state, i+1, one_versus_all=self.one_versus_all)
            snake_health_i = info["snake_health"][i]
            episode = info["episodes"]
            action = agent.act(state_agent_i, i, turn_count, snake_health_i,
                               episode, eps)
            actions[i] = action

        return actions

    def step(self, state, actions, reward, next_state, dones, info, turn_count,
             should_learn):
        for i, agent in agents.items():
            if i not in actions:
                continue
            state_agent_i = sort_states_for_snake_id(
                state, i+1, one_versus_all=self.one_versus_all)
            next_state_agent_i = sort_states_for_snake_id(
                next_state, i+1, one_versus_all=self.one_versus_all)
            
            snake_health_i = info["snake_health"][i]
            episode = info["episodes"]

            agent.step(state_agent_i, actions[i], reward[i],
                       next_state_agent_i, dones[i], snake_id=i,
                       turn_count=turn_count,
                       snake_health=snake_health_i,
                       episode=episode,
                       should_learn=should_learn)

    def save(self, name, i_episode):
        if self.models_to_save == "all":
            agents[0].qnetwork_local.save_parameters(
                    '{}/local-{}-e{}.params'.format(
                        self.model_dir, name, i_episode))
            agents[0].qnetwork_local.export(
                    '{}/local-{}-e{}'.format(
                        self.model_dir, name, i_episode))
            agents[0].qnetwork_target.save_parameters(
                    '{}/target-{}-e{}.params'.format(
                        self.model_dir, name, i_episode))
            agents[0].qnetwork_target.export(
                    '{}/target-{}-e{}'.format(
                        self.model_dir, name, i_episode))
        else:
            agents[0].qnetwork_local.export('{}/local'.format(
                        self.model_dir))

    def reset(self):
        for _, agent in agents.items():
            agent.reset()

class Agent:
    '''
    Agent that interacts and learns from the environment
    '''
    def __init__(self, agent_id, state_shape,
                 action_size, seed,
                 sequence_length,
                 buffer_size, update_every,
                 lr_start, lr_step, lr_factor,
                 gamma, tau, batch_size,
                 qnetwork_local, qnetwork_target):
        
        self.action_size = action_size
        self.sequence_length = sequence_length
        self.seed = random.seed(seed)
        self.update_every = update_every
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

        self.qnetwork_local = qnetwork_local
        self.qnetwork_target = qnetwork_target
        
        self.schedule = mx.lr_scheduler.FactorScheduler(step=lr_step,
                                                        factor=lr_factor)
        self.schedule.base_lr = lr_start
        adam_optimizer = mx.optimizer.Adam(learning_rate=lr_start,
                                           lr_scheduler=self.schedule)

        self.local_model_params = self.qnetwork_local.collect_params()
        self.target_model_params = self.qnetwork_target.collect_params()

        self.trainer = gluon.Trainer(self.local_model_params,
                                     optimizer=adam_optimizer)

        self.memory = ReplayBuffer(agent_id, action_size, buffer_size,
                                   self.batch_size, seed)

        self.loss_function = gluon.loss.L2Loss()
        self.done = False

    def act(self, state, snake_id, turn_count, snake_health, episode, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            snake_id (int): ID of the snake
            turn_count (int): turn count of the game
            snake_health (int): health of the snake
            episode (int): The current episode, used for checking previous states
            eps (float): epsilon, for epsilon-greedy action selection
        """
        # Epsilon-greedy action selection
        if random.random() > eps:
            snake_health = snake_health - 1 # Account for taking the current move

            empty_state = np.zeros(state.shape)
            turn_count_eos = -1
            snake_health_eos = 101

            with autograd.predict_mode():
                last_n_memory = self.memory.get_last_n(n=self.sequence_length - 1)
                state_sequence, snake_id_sequence = [], []
                turn_count_sequence, snake_health_sequence = [], []
                for i in range(self.sequence_length):
                    if i == self.sequence_length - 1:
                        turn_count_i = turn_count
                        episode_i = episode
                        delta = 0
                        state_i = state
                        snake_health_i = snake_health
                    else:
                        turn_count_i = last_n_memory[i].turn_count
                        episode_i = last_n_memory[i].episode
                        delta = self.sequence_length - 1 - i
                        state_i =  last_n_memory[i].state
                        snake_health_i = last_n_memory[i].snake_health

                    episode_correct = episode_i == episode
                    turn_correct = turn_count_i + delta == turn_count
                    if episode_correct and turn_correct:
                        state_sequence.append(state_i)
                        turn_count_sequence.append(turn_count_i)
                        snake_health_sequence.append(snake_health_i)
                    else:
                        state_sequence.append(empty_state)
                        turn_count_sequence.append(turn_count_eos)
                        snake_health_sequence.append(snake_health_eos)

                state_sequence = mx.nd.array(np.stack(state_sequence),
                                             ctx=ctx).transpose((0, 3, 1, 2)).expand_dims(0)
                turn_count_sequence = mx.nd.array(np.stack(turn_count_sequence),
                                                  ctx=ctx).expand_dims(0)
                snake_health_sequence = mx.nd.array(np.stack(snake_health_sequence),
                                                  ctx=ctx).expand_dims(0)

                snake_id_sequence = mx.nd.array(np.array([snake_id]*self.sequence_length),
                                                ctx=ctx).expand_dims(0)
                
                if self.qnetwork_local.take_additional_forward_arguments:
                    action_values = self.qnetwork_local(state_sequence,
                                                        snake_id_sequence,
                                                        turn_count_sequence,
                                                        snake_health_sequence)
                else:
                    action_values = self.qnetwork_local(state_sequence)

            return np.argmax(action_values.asnumpy())
        else:
            last_memory = self.memory.get_last_n(n=1)[0]
            if last_memory is not None:
                # Disable choosing random actions of forbiden moves
                last_action = last_memory.action
                last_episode = last_memory.episode
                last_turn_count = last_memory.turn_count
                if last_episode == episode and (last_turn_count == turn_count - 1):
                    # Check that last_memory is from the same episode
                    action_space = [*range(self.action_size)]
                    action_space.remove(last_action)
                    return random.choice(action_space)
            return random.choice(np.arange(self.action_size))

    def step(self, state, action, reward, next_state, done, snake_id, turn_count,
             snake_health, episode, should_learn):
        if self.done:
            return

        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done, snake_id,
                        turn_count, snake_health, episode)

        if done:
            self.done = True

        if not should_learn:
            return
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size*3:
                experiences = self.memory.sample(self.sequence_length)
                self.learn(experiences, self.gamma)

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences: tuple of (s, a, r, s', done, snake_id, turn_count, snake_health) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, snake_id, turn_count, snake_health = experiences
        
        # Get max predicted Q values (for next states) from target model
        with autograd.predict_mode():
            if self.qnetwork_target.take_additional_forward_arguments:
                Q_targets_next = self.qnetwork_target(next_states, snake_id, turn_count, snake_health
                                                 ).max(1).expand_dims(1)
            else:
                Q_targets_next = self.qnetwork_target(next_states).max(1).expand_dims(1)
        
        # Compute Q targets for current states
        dones = dones.astype(np.float32)
        Q_targets = rewards[:, -1].expand_dims(1) + (gamma * Q_targets_next * (1 - dones[:, -1].expand_dims(1)))

        # Get expected Q values from local model
        last_action = actions[:, -1].expand_dims(1)
        action_indices = nd.array(np.arange(0, last_action.shape[0])).as_in_context(ctx)
        action_indices.attach_grad()
        last_actions = nd.concat(action_indices.expand_dims(1), last_action, dim=1)
        
        with autograd.record():
            if self.qnetwork_local.take_additional_forward_arguments:
                predicted_actions = self.qnetwork_local(states, snake_id, turn_count, snake_health)
            else:
                predicted_actions = self.qnetwork_local(states)

            Q_expected = nd.gather_nd(predicted_actions, last_actions.T)

            # Compute loss
            loss = self.loss_function(Q_expected, Q_targets)
        
        # Minimize the loss
        loss.backward()
        self.trainer.step(Q_expected.shape[0])
        
        # ------------------- update target network ------------------- #
        self.soft_update(self.tau)
        
    def soft_update(self, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_name, local_name in zip(self.target_model_params, self.local_model_params):
            target_params = self.target_model_params[target_name]
            local_params = self.local_model_params[local_name]
            target_params.set_data(tau*local_params.data() + (1.0 - tau)*target_params.data())
            
    def reset(self):
        self.done = False

class ReplayBuffer:
    ACTION_ZEROPAD = 4
    def __init__(self, agent_id, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            agent_id (int)
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.agent_id = agent_id
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.learning_count = 0
        self.pool = None
        self.experience = namedtuple("Experience",
                                     field_names=["state", "action", "reward",
                                                  "next_state", "done", "snake_id",
                                                  "turn_count", "snake_health",
                                                  "episode"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done, snake_id, turn_count,
            snake_health, episode):
        # print("Added ID {} episode {} turn {} snake_health {} ".format(
        #     snake_id, episode, turn_count, snake_health))
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done, snake_id,
                            turn_count, snake_health, episode)
        self.memory.append(e)
        
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def clear(self):
        self.memory.clear()

    def sample_from_index(agent_id, sequence_length, index):
        memory = agents[agent_id].memory.memory
        
        states_i, actions_i, rewards_i, next_states_i = [], [], [], []
        dones_i, snake_id_i, turn_count_i, snake_health_i = [], [], [], []
        experience_0 = memory[index]
        turn_0 = experience_0.turn_count
        episode_0 = experience_0.episode

        count = 0
        for i in range(sequence_length):
            experience_i = memory[index - i]
            turn_i = experience_i.turn_count
            episode_i = experience_i.episode
            
            episode_correct = episode_i == episode_0
            turn_count_correct = turn_i == turn_0 - i

            last_turn_count = 0
            if episode_correct and turn_count_correct:
                count += 1
                states_i.insert(0, experience_i.state.transpose((2, 0, 1)))
                actions_i.insert(0, experience_i.action)
                rewards_i.insert(0, experience_i.reward)
                next_states_i.insert(0,
                                     experience_i.next_state.transpose((2, 0, 1)))
                dones_i.insert(0, experience_i.done)
                snake_id_i.insert(0, experience_i.snake_id)
                turn_count_i.insert(0, experience_i.turn_count)
                snake_health_i.insert(0, experience_i.snake_health)
                
        # Zero pad all experiences to sequence_length
        empty_state = np.zeros(states_i[-1].shape)
        last_snake_id = snake_id_i[-1]
        for _ in range(sequence_length - count):
            states_i.insert(0, empty_state)
            actions_i.insert(0, ReplayBuffer.ACTION_ZEROPAD)
            rewards_i.insert(0, 0)
            next_states_i.insert(0, empty_state)
            dones_i.insert(0, 0)
            snake_id_i.insert(0, last_snake_id)
            turn_count_i.insert(0, -1)
            snake_health_i.insert(0, 101)

        # Stack experiences
        states_i = np.stack(states_i)
        actions_i = np.stack(actions_i)
        rewards_i = np.stack(rewards_i)
        next_states_i = np.stack(next_states_i)
        dones_i = np.stack(dones_i)
        snake_id_i = np.stack(snake_id_i)
        turn_count_i = np.stack(turn_count_i)
        snake_health_i = np.stack(snake_health_i)
        
        return (states_i, actions_i, rewards_i, next_states_i, dones_i, snake_id_i,
                turn_count_i, snake_health_i)

    def sample(self, sequence_length):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        snake_id, turn_count, snake_health = [], [], []

        """Randomly sample a batch of experiences from memory."""
        indexes = random.sample(range(sequence_length, len(self.memory)-1), k=self.batch_size)

        for index in indexes:
            states_i, actions_i, rewards_i, next_states_i, dones_i, snake_id_i, turn_count_i, snake_health_i = ReplayBuffer.sample_from_index(
                self.agent_id, sequence_length, index)
            states.append(states_i)
            actions.append(actions_i)
            rewards.append(rewards_i)
            next_states.append(next_states_i)
            dones.append(dones_i)
            snake_id.append(snake_id_i)
            turn_count.append(turn_count_i)
            snake_health.append(snake_health_i)

        states = np.stack(states)
        actions = np.stack(actions)
        rewards = np.stack(rewards)
        next_states = np.stack(next_states)
        dones = np.stack(dones)
        snake_id = np.stack(snake_id)
        turn_count = np.stack(turn_count)
        snake_health = np.stack(snake_health)

        states = mx.nd.array(states, ctx=ctx).astype(
            np.float32, copy=False)
        actions = mx.nd.array(actions, ctx=ctx).astype(
            np.float32, copy=False)
        rewards = mx.nd.array(rewards, ctx=ctx).astype(
            np.float32, copy=False)
        next_states = mx.nd.array(next_states, ctx=ctx).astype(
            np.float32, copy=False)
        dones = mx.nd.array(dones, ctx=ctx).astype(
            np.float32, copy=False)
        snake_id  = mx.nd.array(snake_id, ctx=ctx).astype(
            np.float32, copy=False)
        turn_count = mx.nd.array(turn_count, ctx=ctx).astype(
            np.float32, copy=False)
        snake_health = mx.nd.array(snake_health, ctx=ctx).astype(
            np.float32, copy=False)

        return (states, actions, rewards, next_states, dones, snake_id,
                turn_count, snake_health)

    def copy_contents_to_other_buffer(self, other_buffer):
        for value in self.memory:
            other_buffer.add(value.state, value.action, value.reward,
                             value.next_state, value.done, value.snake_id,
                             value.turn_count, value.snake_health)

    def get_last_n(self, n):
        last_n = []
        for i in range(n):
            try:
                value = self.memory[i - n]
            except IndexError:
                value = None

            last_n.append(value)
        return last_n
