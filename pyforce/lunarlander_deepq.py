import tensorflow as tf
import numpy as np
import core as pf
import time
import gym
import sys

env = gym.make('CartPole-v0')
net = pf.Network(network={'type': 'fc', 'layers': [4, 256, 256, 2]},
                 agent='deepq',
                 actions=[0, 1],
                 learning_rate=0.0003,
                 decay_steps=1000,
                 decay_factor=0.99,
                 keep_prob=0.50,
                 batch_size=32)
net.build()
agent = pf.DeepQ(net=net,
                 env=env,
                 discount_rate=1.00,
                 max_episodes=500,
                 score=np.sum,
                 min_mem=10 ** 4,
                 max_mem=10 ** 5,
                 seq_length=1,
                 batch_size=32,
                 print_episodes=10,
                 decay_episodes=1,
                 decay_factor=0.99,
                 clone_steps=1000,
                 init_epsilon=0.50,
                 min_epsilon=0.01)
agent.train()
agent.epsilon = agent.min_epsilon
agent.evaluate()
