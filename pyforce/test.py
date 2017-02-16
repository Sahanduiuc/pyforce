import tensorflow as tf
import numpy as np
import core as pf
import gym
import sys

batch_size = 100
seq_length = 1

net = pf.Network(network={'type': 'fc', 'layers': [8 * seq_length, 64, 64, 4]},
                 agent='deepq',
                 actions=[0, 1, 2, 3],
                 learning_rate=0.001,
                 decay_steps=100,
                 decay_factor=0.99,
                 keep_prob=1.00,
                 batch_size=batch_size)

env = gym.make('LunarLander-v2')
def score(rewards):
    return np.sum(rewards)
agent = pf.DeepQ(net=net,
                 env=env,
                 discount_rate=0.99,
                 max_episodes=500,
                 score=score,
                 max_mem=1000,  # this is a bottle-neck! re-write!
                 seq_length=seq_length,
                 batch_size=batch_size,
                 gamma=1.00)
agent.train()
