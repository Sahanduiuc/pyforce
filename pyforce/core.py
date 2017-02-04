import tensorflow as tf
import h5py as h5
import numpy as np
import gym

# TODO: How to deal with action spaces that depends on the state?
# TODO: Let the network keep track of alpha?

def gain(rewards, start=0, gamma=1.0):
    R = rewards[start + 1:]
    d = gamma ** np.arange(0, len(R))
    return np.sum(R * d)

def progress(current, maximum, step_ratio=0.05):
    check = current % int(maximum * step_ratio)
    if check == 0:
        stars = int(current / int(maximum * 0.05))
        spaces = int(1 / step_ratio) - stars
        if stars < int(1 / step_ratio):
            bar = "\tProgress: " + "#" * stars + " " * spaces + " ({:.2f})"
            ratio = stars / int(1 / step_ratio)
            print(bar.format(ratio), end="\r")
        else:
            print("\tProgress: " + "#" * stars + " (1.00) - Done", end="\n")

class Network():
    """A TensorFlow neural network with methods for reinforcement learning.

       The network should be thought of as a network for classification with N classes.
       In this case the N classes are N possible actions in the action space.
    """

    def __init__(self, eval_op, weights, actions, gradient, update_op, session, placeholders, init):
        self.eval_op = eval_op
        self.weights = weights
        self.actions = actions
        self.gradient = gradient
        self.update_op = update_op
        self.session = session
        self.placeholders = placeholders  # {'state_pl': state_pl, ..., 'alpha_pl': alpha_pl}
        self.init = init

    def __str__(self):
        return "Network()"

    def __repr__(self):
        return "<pf.Network>"

    def initialize(self):
        self.session.run([self.init])

    def evaluate(self, s, a=None):
        """Network evaluated at s, then a."""


        state_pl = self.placeholders['state_pl']
        output = self.session.run(self.eval_op, feed_dict={state_pl:s})
        if a is None:
            return output
        else:
            a_idx = self.actions.index(a)
            return output[0, a_idx]

    def differentiate(self, s, a):
        """Gradient of the network evaluated at s, then a."""
        state_pl = self.placeholders['state_pl']
        g = self.session.run(self.gradient, feed_dict={state_pl:s})
        return g[self.actions.index(a)]

    def update(self, a, s, g, p, alpha):
        a_idx = self.actions.index(a)
        feed_dict = {self.placeholders['state_pl']:s,
                     self.placeholders['gain_pl']:g,
                     self.placeholders['value_pl']:p,
                     self.placeholders['alpha_pl']:alpha}
        self.session.run(self.update_op[a_idx], feed_dict=feed_dict)

    def choice(self, s):
        """Sample an action from probabilities given by forward pass of s."""
        a = np.array(self.actions)
        k, = a.shape
        return np.random.choice(a, p=self.evaluate(s).reshape(k,))

    def get_weights(self):
        return self.session.run(self.weights)

    def set_weights(self):
        pass

class Learner():
    """Combines network and environment to implement training."""

    def __init__(self, network, env, max_episodes, alpha, discount=1.0):
        self.network = network  # a Keras model
        self.env = env  # a Gym environment
        self.max_episodes = max_episodes
        self.alpha = alpha
        self.discount = discount
        self.states = []
        self.actions = []
        self.rewards = []
        self.episodes = 0

    def __str__(self):
        s = format(env).rstrip(" instance>") + ">"
        return "Learner for environment {}.".format(s)

    def __repr__(self):
        s = format(env).rstrip(" instance>") + ">"
        return "Learner(environment={})".format(s)

    def step(self):
        pass

    def train(self):
        pass

class Reinforce(Learner):

    def __init__(self, network, env, max_episodes, alpha, discount=1.0):
        Learner.__init__(self, network, env, max_episodes, alpha, discount)

    def __str__(self):
        s = format(env).rstrip(" instance>") + ">"
        return "REINFORCE Learner for environment {}.".format(s)

    def __repr__(self):
        s = format(env).rstrip(" instance>") + ">"
        return "REINFORCE Learner(environment={})".format(s)

    def step(self):
        """Perform one step of the learning algorithm (one episode for REINFORCE)."""

        self.env.reset()
        self.states = []
        self.actions = []
        self.rewards = []
        s = self.env.state
        done = False
        steps = 0
        while not done:
            self.states.append(s)
            a = self.network.choice(s)
            s, r, done, info = self.env.step(a)
            steps += 1
            self.actions.append(a)
            self.rewards.append(r)
            if done:
                self.episodes += 1
        for t in np.arange(0, steps):
            s = self.states[t]
            a = self.actions[t]
            a_idx = self.network.actions.index(a)
            g = gain(self.rewards, start=t, gamma=self.discount)
            p = self.network.evaluate(s, a)
            self.network.update(a, s, g, p, self.alpha * self.discount ** t)

    def train(self):
        self.episodes = 0
        while self.episodes < self.max_episodes:
            self.step()
            progress(self.episodes, self.max_episodes)

class Sarsa(Learner):

    def __init__(self, net, environment):
        self.init(net, environment)
        self.alpha = 1.0

    def __str__(self):
        s = format(env).rstrip(" instance>") + ">"
        return "Sarsa Learner for environment {}.".format(s)

    def __repr__(self):
        s = format(env).rstrip(" instance>") + ">"
        return "Sarsa Learner(environment={})".format(s)

    def step(self):
        pass

    def train(self):
        pass
