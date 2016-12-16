from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization
from keras.regularizers import l2
from keras.optimizers import RMSprop
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

class Environment():
    """Wrapper class for OpenAI Gym environments with convenience methods."""
    def __init__(self, name, env):
        self.name = name
        self.env = env  # gym.Env
        self.env.reset()  # make sure it's reset!

    def run_episode(self, actions=None):
        episode = Episode()
        done = False
        episode.states.append(env.state)
        while not done:
            if actions == None:
                a = self.env.action_space.sample()
            else:
                a = actions(self.env.state).sample()  # pi(s, theta)
            s, r, done, info = self.env.step(a)
            episode.actions.append(a)
            episode.rewards.append(r)
            if not done:
                episode.states.append(s)
        self.env.reset()
        return episode

class Network():
    """
    Neural network specific to policy and value gradient algorithms.

    A wrapper for a collection of TensorFlow operations. You should first
    define the operations describing the architecture of the network, then
    initialize the network from these operations.

    """
    def __init__(self, inference, loss, train_op, eval_op):
        # self.sess = tf.Session()
        self.features_pl = None
        self.labels_pl = None
        self.inference = None
        self.loss = None
        self.train_op = None
        self.eval_op = None

    def __repr__(self):
        pass

    def logits(self):
        return self.inference(features_pl, dropout=True)

    def predictions(self):
        return self.inference(features_pl, dropout=False)

class Learner():
    """Combines network and environment and implements training."""

    def __init__(self, network, environment):
        self.network = network  # a Keras model
        self.environment = environment  # a Gym environment

    def __str__(self):
        return "Learner using {} method for environment {}.".format(self.method, self.environment.name)

    def __repr__(self):
        return "Learner(method={}, environment={})".format(self.method, self.environment.name)

    def step(self):
        """Complete a single episode of parameter updates."""
        episode = self.environment.run_episode()
        T = len(episode)
        for t in range(0, T):
            G = episode.gain(start=t)
            self.network.update(G, episode.states[t], episode.actions[t])

    def train(self):
        """Train the learner.

        Combines TensorFlow training loop with Gym learning loops using network
        and environment objects.

        """
        while episodes < MAXEPISODES:
            episode = self.environment.run_episode()
            episodes += 1
            T = len(episode)
            for t in range(0, T):
                G = episode.gain(start=t)
                self.network.update(G, episode.states[t], episode.actions[t])

class Reinforce(Learner):

    def __init__(self, network, environment):
        self.init(network, environment)
        self.alpha = 1.0

    def step(self):
        pass

    def train(self):
        pass

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
