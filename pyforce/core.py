import tensorflow as tf
import h5py as h5
import numpy as np
import gym

# NOTE:

# TODO:

NN = 512
NUM_FEATS = 4
NUM_ACTIONS = 2
MAX_EPISODES = 500
LEARNING_RATE = 0.0001
DECAY_STEPS = 1000
DECAY_FACTOR = 0.90

def placeholders():
    num_features = 4
    num_actions = 2
    state_pl = tf.placeholder(tf.float32, shape=(1, num_features))
    action_pl = tf.placeholder(tf.int32, shape=(num_actions, 1))
    gain_pl = tf.placeholder(tf.float32, shape=())
    prob_pl = tf.placeholder(tf.float32, shape=())
    return state_pl, action_pl, gain_pl, prob_pl

def activation(features, keep_prob):
    """Returns probability of each action."""
    nn = NN
    num_features = NUM_FEATS
    num_actions = NUM_ACTIONS
    with tf.name_scope('full'):
        sd = 1.0 / np.sqrt(float(num_features))
        weights = tf.Variable(tf.truncated_normal([num_features, nn], stddev=sd), name='weights')
        bias = tf.Variable(tf.constant(0.1, shape=[nn]), name='bias')
        full =  tf.nn.dropout(tf.nn.relu(tf.matmul(features, weights) + bias), keep_prob)
    with tf.name_scope('softmax'):
        sd = 1.0 / np.sqrt(float(nn))
        weights = tf.Variable(tf.truncated_normal([nn, num_actions], stddev=sd), name='weights')
        bias = tf.Variable(tf.constant(0.0, shape=[num_actions]), name='bias')
        logits =  tf.matmul(full, weights) + bias
    return logits

def eligibility(logits, action, gain):
    eligibility = tf.log(tf.matmul(tf.nn.softmax(logits), tf.cast(action, tf.float32)))
    loss = - gain * eligibility
    # tf.summary.scalar('eligibility', eligibility)
    return loss

def update(loss, lr, global_step, decay_steps, decay_factor):
    """Peforms modified (RMSProp) REINFORCE update."""
    lr = tf.train.exponential_decay(lr, global_step, decay_steps, decay_factor)
    # tf.summary.scalar('learning_rate', lr)
    optimizer = tf.train.GradientDescentOptimizer(lr)
    return optimizer.minimize(loss, global_step=global_step)

class Network():

    def __init__(self, actions, keep_prob):
        self.actions = actions
        self.keep_prob = keep_prob
        with tf.Graph().as_default():
            self.global_step = tf.Variable(0, trainable=False, name="global_step")
            self.features_pl = tf.placeholder(tf.float32, shape=(1, 4))
            self.action_pl = tf.placeholder(tf.float32, shape=(2, 1))
            self.gain_pl = tf.placeholder(tf.float32, shape=())
            self.prob_pl = tf.placeholder(tf.float32, shape=())
            self.logits = activation(self.features_pl, self.prob_pl)
            self.loss = eligibility(self.logits, self.action_pl, self.gain_pl)
            self.train_op = update(self.loss, LEARNING_RATE, self.global_step, DECAY_STEPS, DECAY_FACTOR)
            self.init_op = tf.global_variables_initializer()
            self.sess = tf.Session()
            # self.summarizer = tf.summary.merge_all()
            # self.saver = tf.train.Saver()
            # self.writer = tf.summary.FileWriter(log_dir, self.sess.graph)
            self.sess.run(self.init_op)

    def __str__(self):
        return "Network()"

    def __repr__(self):
        return "<pf.Network>"

    def close(self):
        self.sess.close()

    def evaluate(self, s, a=None):
        """Network evaluated at s, then (optionally) at a."""
        # state_pl = self.placeholders['state_pl']
        output = self.sess.run(self.logits, feed_dict={self.features_pl:s,
                                                       self.prob_pl:self.keep_prob})
        if a is None:
            return output
        else:
            a_idx = self.actions.index(a)
            return output[0, a_idx]

    def train(self, s, a, g):
        feed_dict = {self.features_pl:s,
                     self.action_pl:a,
                     self.gain_pl:g,
                     self.prob_pl:self.keep_prob}
        self.sess.run(self.train_op, feed_dict=feed_dict)

    def choice(self, s):
        """Sample an action from probabilities given by forward pass of s."""
        y = self.evaluate(s)
        p = np.exp(y) / np.sum(np.exp(y))
        return np.random.choice(self.actions, p=p.reshape(len(self.actions),))

    def probabilities(self, s):
        y = self.evaluate(s)
        return np.exp(y) / np.sum(np.exp(y))

def gain(rewards, start=0, gamma=1.0):
    R = rewards[start + 1:]
    d = gamma ** np.arange(0, len(R))
    return np.sum(R * d)

def progressBar(current, maximum, step_ratio=0.05):
    check = current % int(maximum * step_ratio)
    if check == 0:
        stars = int(current / int(maximum * 0.05))
        spaces = int(1 / step_ratio) - stars
        if stars < int(1 / step_ratio):
            bar = "Progress: " + "#" * stars + " " * spaces + " ({}/{})"
            print(bar.format(current, maximum), end="\r")
        else:
            bar = "Progress: " + "#" * stars + " ({}/{}) - Done"
            print(bar.format(current, maximum), end="\n")

class Learner():
    """Combines network and environment to implement training."""

    def __init__(self, net, env, discount_rate, score):
        self.net = net  # a Keras model
        self.env = env  # a Gym environment
        self.discount_rate = discount_rate
        self.score = score
        self.scores = []
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

    def __init__(self, net, env, discount_rate, score):
        Learner.__init__(self, net, env, discount_rate, score)

    def __str__(self):
        s = format(env).rstrip(" instance>") + ">"
        return "REINFORCE Learner for environment {}.".format(s)

    def __repr__(self):
        s = format(env).rstrip(" instance>") + ">"
        return "REINFORCE Learner(environment={})".format(s)

    def step(self):
        """Perform one step of the learning algorithm (one episode for REINFORCE)."""

        # Generate an episode
        self.env.reset()
        self.states = []
        self.actions = []
        self.rewards = []
        s = self.env.state
        done = False
        steps = 0
        while not done:
            self.states.append(s)
            a = self.net.choice(s.reshape(1,4))
            s, r, done, info = self.env.step(a)
            steps += 1
            self.actions.append(a)
            self.rewards.append(r)
            if done:
                self.episodes += 1

        # Perform REINFORCE updates
        for t in np.arange(0, steps):
            s = self.states[t]
            a_idx = self.net.actions.index(self.actions[t])
            a = np.zeros((len(self.net.actions), 1))
            a[a_idx, 0] = 1
            g = gain(self.rewards, start=t, gamma=self.discount_rate) * self.discount_rate ** t
            self.net.train(s.reshape(1,4), a, g)

        return self.score((self.rewards))

    def train(self):
        self.episodes = 0
        self.scores = []
        progressBar(self.episodes, MAX_EPISODES)
        while self.episodes < MAX_EPISODES:
            score = self.step()
            self.scores.append(score)
            progressBar(self.episodes, MAX_EPISODES)

    def summary(self):
        if len(self.scores) > 0:
            x = np.array(self.scores)
            x.describe()

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
