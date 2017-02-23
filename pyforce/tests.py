import tensorflow as tf
import numpy as np
import core as pf
import time
import gym
import sys

env = gym.make('CartPole-v0')
for lr in 10.0 ** np.random.uniform(-3, -5, size=10):
    print("Training with lr={:f}".format(lr))
    net = pf.Network(network={'type': 'fc', 'layers': [4 * 1, 256, 256, 2]},
                     agent='deepq',
                     actions=[0, 1],
                     learning_rate=lr,
                     decay_steps=1000,
                     decay_factor=0.99,
                     keep_prob=0.50,
                     batch_size=32)
    net.build()
    agent = pf.DeepQ(net=net,
                     env=env,
                     discount_rate=0.99,
                     max_episodes=500,
                     score=np.sum,
                     min_mem=10 ** 4,  # 5-10% of max_mem
                     max_mem=10 ** 5,
                     seq_length=1,
                     batch_size=32,
                     print_episodes=10,
                     decay_episodes=5,  # min_epsilon @ 5-10% of max_episodes
                     decay_factor=0.9,
                     clone_steps=1000,  # 1% of max_steps
                     init_epsilon=1.00,  # 1.00
                     min_epsilon=0.05)  # 0.10
    agent.train()
