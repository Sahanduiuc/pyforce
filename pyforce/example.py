from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l1, l2
from keras.optimizers import SGD

import numpy as np
import pyforce as pf
import gym

p = 0.50
alpha = 0.10
eta = 0.001

# Model
model = Sequential()
model.add(Dense(...))
model.add(Dropout(p))
model.add(Dense(...))
model.add(Dropout(p))
model.add(Dense(..)))

# Optimizer
optimizer = SGD()

# Environment
env = gym.make('CartPole-v0')

# Learner
learner = pf.Reinforce(model=model, env=env)
learner.train(episodes=episodes)
