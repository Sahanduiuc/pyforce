from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.utils.io_utils import HDF5Matrix
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
import h5py as h5
import numpy as np
import gym

class Episode():
    """Data structure to keep track of learning episode."""

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def gain(self, start, discount):
        R = np.array(self.rewards)[start:]
        d = discount ** np.arange(1, len(R) + 1)
        return np.sum(R * d)


# Not sure that we really need this...
# class Environment():
#     """Wrapper class for OpenAI Gym environments with convenience methods."""
#     def __init__(self, name, env):
#         self.name = name
#         self.env = env  # gym.Env
#         self.env.reset()  # make sure it's reset!
#
#     def run_episode(self, actions=None):
#         episode = Episode()
#         done = False
#         episode.states.append(env.state)
#         while not done:
#             if actions == None:
#                 a = self.env.action_space.sample()
#             else:
#                 a = actions(self.env.state).sample()  # pi(s, theta)
#             s, r, done, info = self.env.step(a)
#             episode.actions.append(a)
#             episode.rewards.append(r)
#             if not done:
#                 episode.states.append(s)
#         self.env.reset()
#         return episode
#
# ... or this...
# class Network():
#     """
#     Neural network specific to policy and value gradient algorithms.
#
#     A wrapper for a collection of TensorFlow operations. You should first
#     define the operations describing the architecture of the network, then
#     initialize the network from these operations.
#
#     """
#     def __init__(self, inference, loss, train_op, eval_op):
#         # self.sess = tf.Session()
#         self.features_pl = None
#         self.labels_pl = None
#         self.inference = None
#         self.loss = None
#         self.train_op = None
#         self.eval_op = None
#
#     def __repr__(self):
#         pass
#
#     def logits(self):
#         return self.inference(features_pl, dropout=True)
#
#     def predictions(self):
#         return self.inference(features_pl, dropout=False)

class Learner():
    """Combines model and environment to implement training."""

    def __init__(self, model, env):
        self.model = model  # a Keras model
        self.env = env  # a Gym environment

    def __str__(self):
        return "Learner using {} method for environment {}.".format(self.method, self.environment.name)

    def __repr__(self):
        return "Learner(method={}, environment={})".format(self.method, self.environment.name)

    def step(self):
        pass

    def train(self):
        pass


class Reinforce(Learner):

    def __init__(self, model, env, max_episodes, baseline=False):
        self.init(model, env)
        self.max_episodes = max_episodes
        self.discount = 1.0

    def step(self):
        state = self.env.state
        while not done:
            action = self.model.predict(state)
            state, reward, done, info = self.env.step(action)
            if done:
                episodes += 1
                print("Episode {} finished in {} steps.".format(episodes, steps))

    def train(self):
        episodes = 0
        while episodes < self.max_episodes:
            episode = Episode()
            state = self.env.reset()
            episode.states += state
            steps = 0
            while not done:
                actions = self.env.action_space
                prob = self.model.predict((actions, state))  # won't work this way: need to evaluate at each action
                action = self.random_action(prob=prob)
                state, reward, done, info = self.env.step(action)
                episode.states += state
                episode.actions += action
                episode.rewards += reward
                steps += 1
                if done:
                    episodes += 1
                    print("Episode {} finished in {} steps.".format(episodes, steps))
                for t in range(0, steps):
                    G = episode.gain(start=t, discount=self.discount)
                    dW = self.model.gradient(state)  # need to implement a gradient method for the model: missing in keras
                    self.model.weights += alpha * G * dW

class Sarsa(Learner):

    def __init__(self, network, environment):
        self.init(network, environment)
        self.alpha = 1.0

    def step(self):
        pass

    def train(self):
        pass

def gain(rewards, start=0, gamma=1.0):
    R = np.array(self.rewards)[start:]
    d = gamma ** np.arange(1, len(rewards[start:]) + 1)
    return np.sum(rewards[start:] * d)
