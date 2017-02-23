import tensorflow as tf
import numpy as np
import core as pf
import time
import gym
import sys


env = gym.make('CartPole-v0')
net = pf.Network(network={'type': 'fc', 'layers': [4, 1024, 512, 2]},
                 agent='reinforce',
                 actions=[0, 1],
                 learning_rate=0.0001,
                 decay_steps=1000,
                 decay_factor=0.90,
                 keep_prob=0.50)
net.build()
agent = pf.Reinforce(net=net,
                     env=env,
                     discount_rate=1.00,
                     max_episodes=500,
                     print_episodes=10,
                     score=np.sum)
agent.train()
agent.evaluate()
