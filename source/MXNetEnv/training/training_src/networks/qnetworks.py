import math

import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon

ctx = mx.gpu(0) if mx.context.num_gpus() > 0 else mx.cpu()

class QNetworkAttention(gluon.nn.HybridBlock):
    def __init__(self, state_shape, action_size,
                 starting_channels,
                 dS,
                 d,
                 number_of_hidden_states,
                 kernel_size,
                 repeat_size,
                 activation_type,
                 sequence_length,
                 seed):
        """Initialize parameters and build model.
        Params
        ======
            state_shape (int, int, int): Dimension of each state
            action_size (int): Dimension of each action
            starting_channels (int):
            dS (int): depth of snake embedding
            d (int): depth of health and turn embedding
            number_of_hidden_states (int)
            repeat_size (int)
            activation_type (str)
            sequence_length (int)
            seed (int): Random seed
        """
        super(QNetworkAttention, self).__init__()

        self.take_additional_forward_arguments = True
        
        self.dS = dS
        self.dH = d
        self.dT = d
        
        self.sequence_length = sequence_length
        self.repeat_size = repeat_size
        mx.random.seed(seed)

        self.conv = gluon.nn.Conv2D(starting_channels,
                                    kernel_size=kernel_size,
                                    strides=2,
                                    activation=activation_type)
        self.conv.collect_params().initialize(mx.init.Xavier(), ctx=ctx)

        self.conv2 = gluon.nn.Conv2D(starting_channels,
                                    kernel_size=kernel_size,
                                    strides=2,
                                    activation=activation_type)
        self.conv2.collect_params().initialize(mx.init.Xavier(), ctx=ctx)

        self.key_norm = gluon.nn.LayerNorm()
        self.key_norm.collect_params().initialize(mx.init.Xavier(), ctx=ctx)

        self.query_norm = gluon.nn.LayerNorm()
        self.query_norm.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
        
        self.conv_snake = gluon.nn.Conv2D(starting_channels, kernel_size=kernel_size,
                                          strides=2)
        self.conv_snake.collect_params().initialize(mx.init.Xavier(), ctx=ctx)

        self.embedding_snake = gluon.nn.Embedding(5, self.dS*starting_channels)
        self.embedding_snake.collect_params().initialize(mx.init.Xavier(), ctx=ctx)

        self.conv_health = gluon.nn.Conv2D(starting_channels, kernel_size=kernel_size,
                                           strides=2)
        self.conv_health.collect_params().initialize(mx.init.Xavier(), ctx=ctx)

        self.embedding_health = gluon.nn.Embedding(100, self.dH*starting_channels)
        self.embedding_health.collect_params().initialize(mx.init.Xavier(), ctx=ctx)

        self.conv_turn = gluon.nn.Conv2D(starting_channels, kernel_size=kernel_size,
                                         strides=2)
        self.conv_turn.collect_params().initialize(mx.init.Xavier(), ctx=ctx)

        self.embedding_turn = gluon.nn.Embedding(10, self.dT*starting_channels)
        self.embedding_turn.collect_params().initialize(mx.init.Xavier(), ctx=ctx)

        self.conv_predict = gluon.nn.Conv2D(starting_channels, kernel_size=kernel_size,
                                             strides=2)
        self.conv_predict.collect_params().initialize(mx.init.Xavier(), ctx=ctx)

        self.predict = gluon.nn.Dense(action_size)
        self.predict.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
   
        if self.sequence_length > 1:
            self.gru = gluon.rnn.GRU(number_of_hidden_states, num_layers=1,
                                     layout='NTC')
            self.gru.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
   
    def hybrid_forward(self, F, state_sequence, snake_id_sequence, turn_count_sequence, snake_health_sequence):
        """Build a network that maps states -> action values."""
        resized_state_sequence = state_sequence.repeat(
                axis=3, repeats=self.repeat_size).repeat(
                    axis=4,repeats=self.repeat_size)
        new_state_sequence = resized_state_sequence.reshape((-3, -2))
        
        states_conv = self.conv(new_state_sequence)

        attentions = [(self.conv_snake, self.embedding_snake, snake_id_sequence,
                       self.dS),
                      (self.conv_turn, self.embedding_turn, turn_count_sequence,
                       self.dT),
                      (self.conv_health, self.embedding_health, snake_health_sequence,
                       self.dH)]

        outs = []
        for conv, embedding, sequence, d in attentions:
            key_external = conv(states_conv).transpose((0, 2, 3, 1))
            reshaped_key_external = key_external.reshape((0, -3, -1))
            reshaped_key_external = self.key_norm(reshaped_key_external)
            
            query_external = embedding(sequence).reshape((-3, int(d), -1))
            query_external = self.query_norm(query_external)
            out = F.linalg_gemm2(reshaped_key_external, query_external,
                                           transpose_b=True, alpha=1/math.sqrt(d))
            outs.append(out)

        outs = F.concat(*outs, dim=2)
        outs_reshaped = F.reshape_like(outs, key_external,
                                       lhs_begin=1, lhs_end=2, rhs_begin=1, rhs_end=3)
        outs_reshaped = outs_reshaped.transpose((0, 3, 1, 2))

        state_value = self.conv2(states_conv)

        concated_with_original = F.concat(state_value, outs_reshaped, dim=1)
        predict = self.conv_predict(concated_with_original)
        predict = predict.reshape((0, -1))
        
        if self.sequence_length > 1:
            predict_reshaped = F.reshape_like(predict, state_sequence,
                                              lhs_begin=0, lhs_end=1, rhs_begin=0, rhs_end=2)
            ts = self.gru(predict_reshaped)
            x = self.predict(ts)
        else:
            x = self.predict(predict)
          
        return x

    def load_only_conv_layers(self, filename):
        load = mx.nd.load(filename)
        layer_key_weight = ['conv.weight', 'conv2.weight',  'conv_snake.weight',
                            'embedding_snake.weight', 'conv_health.weight',
                            'embedding_health.weight', 'conv_turn.weight',
                            'embedding_turn.weight', 'conv_predict.weight']
        layer_weight = [self.conv, self.conv2, self.conv_snake,
                        self.embedding_snake, self.conv_health,
                        self.embedding_health, self.conv_turn,
                        self.embedding_turn, self.conv_predict]
        
        layer_key_bias = ['conv.bias', 'conv2.bias', 'conv_snake.bias', 'conv_health.bias',
                          'conv_turn.bias', 'conv_predict.bias']
        layer_bias = [self.conv, self.conv2, self.conv_snake,
                      self.conv_health, self.conv_turn,
                      self.conv_predict]
        for key, layer in zip(layer_key_weight, layer_weight):
            layer.weight.set_data(load[key])

        for key, layer in zip(layer_key_bias, layer_bias):
            layer.bias.set_data(load[key])

class QNetworkConcat(gluon.nn.HybridBlock):
    def __init__(self, state_shape, action_size,
                 starting_channels,
                 number_of_conv_layers,
                 number_of_dense_layers,
                 number_of_hidden_states,
                 kernel_size,
                 repeat_size,
                 activation_type,
                 sequence_length,
                 seed):
        """Initialize parameters and build model.
        Params
        ======
            state_shape (int, int, int): Dimension of each state
            action_size (int): Dimension of each action
            starting_channels (int):
            number_of_conv_layers (int)
            number_of_dense_layers (int)
            number_of_hidden_states (int)
            repeat_size (int)
            activation_type (str)
            sequence_length (int)
            seed (int): Random seed
        """
        super(QNetworkConcat, self).__init__()
        self.take_additional_forward_arguments = True

        self.sequence_length = sequence_length
        self.repeat_size = repeat_size
        mx.random.seed(seed)
        self.net = gluon.nn.HybridSequential()
        with self.net.name_scope():
            for i in range(number_of_conv_layers):
                self.net.add(gluon.nn.Conv2D(starting_channels*(i+1),
                                             kernel_size=kernel_size,
                                             strides=2,
                                             activation=activation_type))
             
            for _ in range(number_of_dense_layers):
                self.net.add(gluon.nn.Dense(number_of_hidden_states,
                                            activation=activation_type)) 

        self.net.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
       
        self.embedding = gluon.nn.Embedding(5, number_of_hidden_states//2)
        self.embedding.collect_params().initialize(mx.init.Xavier(), ctx=ctx)

        self.combine_net = gluon.nn.HybridSequential()
        self.combine_net.add(gluon.nn.Dense(number_of_hidden_states, activation=activation_type))
        self.combine_net.collect_params().initialize(mx.init.Xavier(), ctx=ctx)

        self.predict = gluon.nn.HybridSequential()
        self.predict.add(gluon.nn.Dense(number_of_hidden_states, activation=activation_type))
        self.predict.add(gluon.nn.Dense(action_size))
        self.predict.collect_params().initialize(mx.init.Xavier(), ctx=ctx)

        if self.sequence_length > 1:
            self.gru = gluon.rnn.GRU(number_of_hidden_states, num_layers=1,
                                     layout='NTC')
            self.gru.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
   
    def hybrid_forward(self, F, state_sequence, snake_id_sequence, turn_count_sequence, snake_health_sequence):
        """Build a network that maps states -> action values."""

        resized_state_sequence = state_sequence.repeat(
                axis=3, repeats=self.repeat_size).repeat(
                    axis=4,repeats=self.repeat_size)
        new_state_sequence = resized_state_sequence.reshape((-3, -2))
        
        if self.sequence_length == 1:
            states_dense = self.net(new_state_sequence).flatten()
            embedded_snake = self.embedding(snake_id_sequence).reshape((-3, -1))
            concatenated = F.concat(states_dense, embedded_snake,
                                    turn_count_sequence, snake_health_sequence,
                                    dim=1)
            x = self.combine_net(concatenated)
        else:
            new_snake_id_embedding = self.embedding(snake_id_sequence.reshape((-1,)))
            new_turn_count_sequence = turn_count_sequence.reshape((-1, 1))
            new_snake_health_sequence = snake_health_sequence.reshape((-1, 1))

            state_dense = self.net(new_state_sequence).flatten()
            concatenated = F.concat(state_dense, new_snake_id_embedding,
                                    new_turn_count_sequence,
                                    new_snake_health_sequence)
            combined = self.combine_net(concatenated)
            states = F.reshape_like(combined, state_sequence,
                                    lhs_begin=0, lhs_end=1, rhs_begin=0, rhs_end=2)
            ts = self.gru(states)
            x = self.predict(ts)
        
        return x
    
class QNetworkVision(gluon.nn.HybridBlock):
    def __init__(self, state_shape, action_size,
                 starting_channels,
                 number_of_conv_layers,
                 number_of_dense_layers,
                 number_of_hidden_states,
                 kernel_size,
                 repeat_size,
                 activation_type,
                 sequence_length,
                 seed):
        """Initialize parameters and build model.
        Params
        ======
            state_shape (int, int, int): Dimension of each state
            action_size (int): Dimension of each action
            starting_channels (int):
            number_of_conv_layers (int)
            number_of_dense_layers (int)
            number_of_hidden_states (int)
            repeat_size (int)
            activation_type (str)
            sequence_length (int)
            seed (int): Random seed
        """
        super(QNetworkVision, self).__init__()
        self.take_additional_forward_arguments = False

        self.sequence_length = sequence_length
        self.repeat_size = repeat_size
        mx.random.seed(seed)
        self.net = gluon.nn.HybridSequential()
        with self.net.name_scope():
            for i in range(number_of_conv_layers):
                self.net.add(gluon.nn.Conv2D(starting_channels*(i+1),
                                             kernel_size=kernel_size,
                                             strides=2,
                                             activation=activation_type))
             
            for _ in range(number_of_dense_layers):
                self.net.add(gluon.nn.Dense(number_of_hidden_states,
                                            activation=activation_type)) 

        self.net.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
       
        self.predict = gluon.nn.HybridSequential()
        self.predict.add(gluon.nn.Dense(number_of_hidden_states, activation=activation_type))
        self.predict.add(gluon.nn.Dense(action_size))
        self.predict.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
   
    def hybrid_forward(self, F, state_sequence):
        """Build a network that maps states -> action values."""

        resized_state_sequence = state_sequence.repeat(
                axis=3, repeats=self.repeat_size).repeat(
                    axis=4,repeats=self.repeat_size)
        new_state_sequence = resized_state_sequence.reshape((-1, -3, -2))
                
        state_dense = self.net(new_state_sequence).flatten()

        x = self.predict(state_dense)

        return x