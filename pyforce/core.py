import tensorflow as tf
import h5py as h5
import numpy as np
import gym

# NOTE:

# TODO: summaries and checkpoints.
# TODO: batch version of updates, using tf.multiply and tf.reduce_mean for loss...

def fc_activation(layers, features, keep_prob):
    """
    Returns probability of each action.

    layers (list): [num_features, nn1, nn2, ..., num_classes]

    """

    num_layers = len(layers)
    full = features
    for i in range(1, num_layers - 1):
        with tf.name_scope('full'):
            sd = 1.0 / np.sqrt(float(layers[i - 1]))
            weights = tf.Variable(tf.truncated_normal([layers[i - 1], layers[i]], stddev=sd), name='weights')
            bias = tf.Variable(tf.constant(0.1, shape=[layers[i]]), name='bias')
            full =  tf.nn.dropout(tf.nn.relu(tf.matmul(full, weights) + bias), keep_prob)
    with tf.name_scope('softmax'):
        sd = 1.0 / np.sqrt(float(layers[-2]))
        weights = tf.Variable(tf.truncated_normal([layers[-2], layers[-1]], stddev=sd), name='weights')
        bias = tf.Variable(tf.constant(0.0, shape=[layers[-1]]), name='bias')
        logits =  tf.matmul(full, weights) + bias
    return logits

def cnn_activation():
    pass

def rnn_activation():
    pass

def reinforce_loss(logits, action, gain):
    p = tf.nn.softmax(logits)
    a = tf.cast(action, tf.float32)
    eligibility = tf.log(tf.reduce_mean(tf.multiply(p, a)))
    loss = - gain * eligibility
    # tf.summary.scaler('loss', loss)
    return loss

def reinforce_update(loss, lr, global_step, decay_steps, decay_factor):
    """Defines REINFORCE update (cf. Sutton & Barto pg. ???)."""
    lr = tf.train.exponential_decay(lr, global_step, decay_steps, decay_factor)
    # tf.summary.scalar('learning_rate', lr)
    optimizer = tf.train.GradientDescentOptimizer(lr)
    return optimizer.minimize(loss, global_step=global_step)

class Network():

    """

    network (dict): {'type': network_type, 'layers': layers_list}
    learner (string): learner_type
    actions (list): [0, ..., num_actions]

    """

    def __init__(self, network, learner, actions, learning_rate, decay_steps,
                 decay_factor, keep_prob=1):
        self.actions = actions
        self.layers = network['layers']
        self.num_features = self.layers[0]
        self.num_classes = self.layers[-1]
        self.keep_prob = keep_prob
        self.learning_rate = learning_rate
        self.decay_steps = decay_steps
        self.decay_factor = decay_factor
        with tf.Graph().as_default():
            self.global_step = tf.Variable(0, trainable=False, name="global_step")
            self.features_pl = tf.placeholder(tf.float32, shape=(1, self.num_features))
            self.action_pl = tf.placeholder(tf.float32, shape=(1, self.num_classes))
            self.prob_pl = tf.placeholder(tf.float32, shape=())
            if learner == 'reinforce':
                self.gain_pl = tf.placeholder(tf.float32, shape=())

            if network['type'] == 'fc':
                self.logits = fc_activation(self.layers, self.features_pl, self.prob_pl)
            elif network['type'] == 'cnn':
                pass
                # self.logits = cnn_activation(self.features_pl, self.prob_pl)
            elif network['type'] == 'rnn':
                pass
                # self.logits = rnn_activation(self.features_pl, self.prob_pl)
            else:
                print("ERROR: Unable to create a network. Network type {} type not recognized.".format(network['type']))

            if learner == "reinforce":
                self.loss = reinforce_loss(self.logits, self.action_pl, self.gain_pl)
                self.train_op = reinforce_update(self.loss,
                                                 self.learning_rate,
                                                 self.global_step,
                                                 self.decay_steps,
                                                 self.decay_factor)
            elif learner == "deepq":
                pass
                # self.loss = deepq_loss(self.logits, self.action_pl)
                # self.train_op = deepq_update(self.loss, self.global_step)
            elif learner == "trpo":
                pass
                # self.loss = trpo_loss(self.logits, self.action_pl)
                # self.train_op = trpo_update(self.loss, self.global_step)
            else:
                print("ERROR: Unable to create a network. Learner type {} type not recognized.".format(learner))

            self.init_op = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(self.init_op)

    def __str__(self):
        return "Network()"

    def __repr__(self):
        return "<pf.Network>"

    def close(self):
        self.sess.close()

    def evaluate(self, s, a=None):
        """Network evaluated at s, then (optionally) at a."""
        feed_dict = {self.features_pl:s.reshape((1, self.num_features)),
                     self.prob_pl:self.keep_prob}
        output = self.sess.run(self.logits, feed_dict=feed_dict)
        if a is None:
            return output
        else:
            a_idx = self.actions.index(a)
            return output[0, a_idx]

    def train(self, s, a, g):
        feed_dict = {self.features_pl:s.reshape(1, self.num_features),
                     self.action_pl:a.reshape(1, self.num_classes),
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
    try:
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
    except:
        print("Error: maximum * step_ratio is non-integer value.")

class Learner():
    """Combines network and environment to implement training."""

    def __init__(self, net, env, discount_rate, max_episodes, score):
        self.net = net  # a Keras model
        self.env = env  # a Gym environment
        self.discount_rate = discount_rate
        self.max_episodes = max_episodes
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

    """

    Parameters
    ----------
    net (): a `pf.Network` object.
    env (): a `gym.Environment` object.
    discount (): the discount rate applied to rewards.
    score (): optional function summarizing the rewards in an epoch.

    """

    def __init__(self, net, env, discount_rate, max_episodes, score=None):
        Learner.__init__(self, net, env, discount_rate, max_episodes, score)

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
            a = np.zeros((1, len(self.net.actions)))
            a[0, a_idx] = 1
            g = gain(self.rewards, start=t, gamma=self.discount_rate) * self.discount_rate ** t
            self.net.train(s.reshape(1,4), a, g)

        return self.score((self.rewards))

    def train(self):
        self.episodes = 0
        self.scores = []
        progressBar(self.episodes, self.max_episodes)
        while self.episodes < self.max_episodes:
            score = self.step()
            self.scores.append(score)
            progressBar(self.episodes, self.max_episodes)

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
        
