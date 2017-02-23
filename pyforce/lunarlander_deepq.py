import tensorflow as tf
import numpy as np
import core as pf
import time
import gym
import sys

discount_rate = 0.99
batch_size = 32
learning_rate = 0.0003
lr_decay_steps = 1000
lr_decay_factor = 0.99
keep_prob = 0.50
min_mem = 10 ** 4
max_mem = 10 ** 5
seq_length = 1
eps_decay_episodes = 1
eps_decay_factor = 0.99
clone_steps = 1000
init_epsilon = 0.50
min_epsilon = 0.01
max_episodes = 500
print_episodes = 10
def score(x):
    return np.sum(x * 0.99 ** np.range(1, len(x) + 1))

env = gym.make('LunarLander-v2')
net = pf.Network(network={'type': 'fc', 'layers': [8, 256, 256, 4]},
                 agent='deepq',
                 actions=[0, 1, 2, 3],
                 learning_rate=learning_rate,
                 decay_steps=lr_decay_steps,
                 decay_factor=lr_decay_factor,
                 keep_prob=keep_prob,
                 batch_size=batch_size)
net.build()
agent = pf.DeepQ(net=net,
                 env=env,
                 discount_rate=discount_rate,
                 max_episodes=max_episodes,
                 score=score,
                 min_mem=min_mem,
                 max_mem=max_mem,
                 seq_length=seq_length,
                 batch_size=batch_size,
                 print_episodes=print_episodes,
                 decay_episodes=eps_decay_episodes,
                 decay_factor=eps_decay_factor,
                 clone_steps=clone_steps,
                 init_epsilon=init_epsilon,
                 min_epsilon=min_epsilon)
agent.train()
agent.epsilon = agent.min_epsilon
agent.evaluate()
