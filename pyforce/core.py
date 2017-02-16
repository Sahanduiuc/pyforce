import tensorflow as tf
import h5py as h5
import numpy as np
import gym

# TODO: check that model and clone output are working (should give different evaluations).
# TODO:

# TODO: summaries and checkpoints.
# TODO: batch version of updates, using tf.multiply and tf.reduce_mean for loss...

def fc_activation(layers, features, keep_prob, clone=False):
    """

    layers (list): [num_features, nn1, nn2, ..., num_classes]
    features (placeholder): [batch_size x num_features]
    clone (boolean): is this a copied network (i.e. for DeepQ training).

    """

    if clone:
        prefix = 'clone_'
    else:
        prefix = 'model_'

    num_layers = len(layers)
    full = features
    for i in range(1, num_layers - 1):
        with tf.name_scope(prefix + 'full'):
            sd = 1.0 / np.sqrt(float(layers[i - 1]))
            weights = tf.Variable(tf.truncated_normal([layers[i - 1], layers[i]], stddev=sd), name='weights')
            bias = tf.Variable(tf.constant(0.1, shape=[layers[i]]), name='bias')
            full =  tf.nn.dropout(tf.nn.relu(tf.matmul(full, weights) + bias), keep_prob)
    with tf.name_scope(prefix + 'softmax'):
        sd = 1.0 / np.sqrt(float(layers[-2]))
        weights = tf.Variable(tf.truncated_normal([layers[-2], layers[-1]], stddev=sd), name='weights')
        bias = tf.Variable(tf.constant(0.0, shape=[layers[-1]]), name='bias')
        logits =  tf.matmul(full, weights) + bias
    return logits

def cnn_activation():
    """
    features (tf.placeholder): [batch_size x height x width x depth]
    """
    pass

def rnn_activation():
    pass

def reinforce_loss(logits, action, gain):
    p = tf.nn.softmax(logits)
    a = tf.cast(action, tf.float32)
    eligibility = tf.log(tf.reduce_sum(tf.multiply(p, a)))  # or reduce_mean?
    loss = - (gain * eligibility)
    tf.summary.scaler('loss', loss)
    return loss

def reinforce_update(loss, lr, global_step, decay_steps, decay_factor):
    """Defines REINFORCE update (cf. Sutton & Barto)."""
    lr = tf.train.exponential_decay(lr, global_step, decay_steps, decay_factor)
    tf.summary.scalar('learning_rate', lr)
    optimizer = tf.train.GradientDescentOptimizer(lr)
    return optimizer.minimize(loss, global_step=global_step)

def deepq_loss(logits, targets, actions):

    """
    Args:
        logits: `Tensor` of shape (batch_size, num_classes).
        targets: `Tensor` of shape (batch_size, ).
    """

    actions = tf.cast(actions, tf.float32)
    logits = tf.reduce_sum(tf.multiply(logits, actions), axis=1)
    mse = tf.reduce_mean(tf.square(targets - logits), axis=0)
    tf.summary.scalar('mse', mse)
    return mse

def deepq_update(loss, lr, global_step, decay_steps, decay_factor):

    """
    Args:
        loss: `Tensor` of shape ().
    """

    lr = tf.train.exponential_decay(lr, global_step, decay_steps, decay_factor)
    tf.summary.scalar('learning_rate', lr)
    optimizer = tf.train.RMSPropOptimizer(lr)
    return optimizer.minimize(loss, global_step=global_step)

def clone_weights(graph):
    weights_to = graph.get_collection('trainable_variables', 'clone')
    weights_from = graph.get_collection('trainable_variables', 'model')
    ops = [tf.assign(w, v) for w,v in zip(weights_to, weights_from)]
    return ops

class Network():

    """

    network (dict): {'type': network_type, 'layers': layers_list}
    agent (string): agent_type
    actions (list): [0, ..., num_actions]

    Note that `layers` takes the form [(input_shape), nn1, nn2, ..., num_classes]. For
    example, if the inputs are vectors, then (input_shape) = num_features. If inputs are
    images, then (input_shape) = (height, width, depth). The form of the values for
    the hidden layers depends on `type`. For fully-connected layers, these values are
    simply integers specifiying the number of neurons in each hidden layer. For
    convolutional networks, the user must specify ... .

    """

    def __init__(self, network, agent, actions, learning_rate, decay_steps,
                 decay_factor, keep_prob=1, batch_size=None):

        self.network = network
        self.agent = agent
        self.actions = actions
        self.learning_rate = learning_rate
        self.decay_steps = decay_steps
        self.decay_factor = decay_factor
        self.keep_prob = keep_prob

        self.type = network['type']
        self.layers = network['layers']
        self.num_features = self.layers[0]
        self.num_classes = self.layers[-1]

        self.graph = tf.Graph()
        if batch_size is not None:
            self.batch_size = batch_size
            if self.agent == 'reinforce':
                assert self.batch_size == 1, "REINFORCE requires that batch_size == 1."

        with self.graph.as_default():
            self.global_step = tf.Variable(0, trainable=False, name="global_step")
            self.features_pl = tf.placeholder(tf.float32, shape=(None, self.num_features))
            self.action_pl = tf.placeholder(tf.float32, shape=(self.batch_size, self.num_classes))
            self.prob_pl = tf.placeholder(tf.float32, shape=())
            if agent == 'reinforce':
                self.gain_pl = tf.placeholder(tf.float32, shape=())
            if agent == 'deepq':
                self.targets_pl = tf.placeholder(tf.float32, shape=(self.batch_size, ))

            if network['type'] == 'fc':
                self.logits = fc_activation(self.layers, self.features_pl, self.prob_pl)
                if agent == 'deepq':
                    self.targets = fc_activation(self.layers, self.features_pl, self.prob_pl, clone=True)
            elif network['type'] == 'cnn':
                pass
                # self.logits = cnn_activation(self.features_pl, self.prob_pl)
            elif network['type'] == 'rnn':
                pass
                # self.logits = rnn_activation(self.features_pl, self.prob_pl)
            else:
                print("ERROR: Unable to create a network. Network type {} type not recognized.".format(network['type']))

            self.softmax = tf.nn.softmax(self.logits)

            if agent == "reinforce":
                self.loss = reinforce_loss(self.logits, self.action_pl, self.gain_pl)
                self.train_op = reinforce_update(self.loss,
                                                 self.learning_rate,
                                                 self.global_step,
                                                 self.decay_steps,
                                                 self.decay_factor)
            elif agent == "deepq":
                self.clone_weights = clone_weights(self.graph)
                self.loss = deepq_loss(self.logits, self.targets_pl, self.action_pl)
                self.train_op = deepq_update(self.loss,
                                             self.learning_rate,
                                             self.global_step,
                                             self.decay_steps,
                                             self.decay_factor)
            elif agent == "trpo":
                pass
                # self.loss = trpo_loss(self.logits, self.action_pl)
                # self.train_op = trpo_update(self.loss, self.global_step)
            else:
                print("ERROR: Unable to create a network. Agent type {} type not recognized.".format(agent))

            self.init_op = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(self.init_op)

    def __str__(self):
        return "Network()"

    def __repr__(self):
        return "<pf.Network>"

    def close(self):
        self.sess.close()

    def evaluate(self, s, a=None, clone=False):
        """Network evaluated at s, then (optionally) at a."""
        # feed_dict = {self.features_pl:s.reshape((self.batch_size, self.num_features)),
        #              self.prob_pl:self.keep_prob}
        feed_dict = {self.features_pl:s.reshape((-1, self.num_features)),
                     self.prob_pl:self.keep_prob}
        if clone:
            output = self.sess.run(self.targets, feed_dict=feed_dict)
        else:
            output = self.sess.run(self.logits, feed_dict=feed_dict)
        if a is None:
            return output
        else:
            a_idx = self.actions.index(a)
            return output[0, a_idx]

    def update(self, states, actions, targets=None, gain=None):
        if self.agent == 'reinforce':
            feed_dict = {self.features_pl:states.reshape(1, self.num_features),
                         self.action_pl:actions.reshape(1, self.num_classes),
                         self.gain_pl:gain,
                         self.prob_pl:self.keep_prob}
            self.sess.run(self.train_op, feed_dict=feed_dict)
        elif self.agent == 'deepq':
            feed_dict = {self.features_pl:states.reshape(self.batch_size, self.num_features),
                         self.action_pl:actions.reshape(self.batch_size, self.num_classes),
                         self.targets_pl:targets,
                         self.prob_pl:self.keep_prob}
            self.sess.run(self.train_op, feed_dict=feed_dict)

    def choice(self, s):
        """Sample an action from probabilities given by forward pass of s."""
        feed_dict = {self.features_pl:s.reshape((1, self.num_features)),
                     self.prob_pl:self.keep_prob}
        p = self.sess.run(self.softmax, feed_dict=feed_dict)
        return np.random.choice(self.actions, p=p.reshape(len(self.actions),))

    def probabilities(self, s):
        feed_dict = {self.features_pl:s.reshape((1, self.num_features)),
                     self.prob_pl:self.keep_prob}
        return self.sess.run(self.softmax, feed_dict=feed_dict)

    def clone(self):
        """Copy the model weights to clone."""
        self.sess.run(self.clone_weights)

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

class Agent():
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
        return "Agent for environment {}.".format(s)

    def __repr__(self):
        s = format(env).rstrip(" instance>") + ">"
        return "Agent(environment={})".format(s)

    def run_episode(self):
        """Inner-loop of learning algorithm."""
        pass

    def train(self):
        """Outer-loop of learning algorithm."""
        pass

    def evaluate(self):
        """Evaluate the performance of the agent."""
        pass

class Reinforce(Agent):

    """

    Parameters
    ----------
    net (): a `pf.Network` object.
    env (): a `gym.Environment` object.
    discount (): the discount rate applied to rewards.
    score (): optional function summarizing the rewards in an epoch.

    """

    def __init__(self, net, env, discount_rate, max_episodes, score=None):
        Agent.__init__(self, net, env, discount_rate, max_episodes, score)

    def __str__(self):
        s = format(env).rstrip(" instance>") + ">"
        return "REINFORCE agent for environment {}.".format(s)

    def __repr__(self):
        s = format(env).rstrip(" instance>") + ">"
        return "REINFORCE agent(environment={})".format(s)

    def run_episode(self):
        """Perform one step of the learning algorithm (one episode for REINFORCE)."""

        # Generate an episode
        self.states = []
        self.actions = []
        self.rewards = []
        s = self.env.reset()
        done = False
        steps = 0
        while not done:
            self.states.append(s)
            a = self.net.choice(s)
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
            self.net.update(s, a, gain=g)

        return self.score((self.rewards))

    def train(self):
        self.episodes = 0
        self.scores = []
        progressBar(self.episodes, self.max_episodes)
        while self.episodes < self.max_episodes:
            score = self.run_episode()
            self.scores.append(score)
            progressBar(self.episodes, self.max_episodes)

class DeepQ(Agent):

    def __init__(self, net, env, discount_rate, max_episodes, score,
                 max_mem, seq_length, batch_size, gamma):
        Agent.__init__(self, net, env, discount_rate, max_episodes, score)
        # self.net_ = net
        self.epsilon = 1.00
        self.min_epsilon = 0.10
        self.memory = []
        self.max_mem = max_mem
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.gamma = gamma
        self.decay_episodes = 100
        self.decay_factor = 0.99
        self.update_steps = 100

    def __str__(self):
        s = format(self.env).rstrip(" instance>") + ">"
        return "Deep-Q agent for environment {}.".format(s)

    def __repr__(self):
        s = format(self.env).rstrip(" instance>") + ">"
        return "Deep-Q agent(environment={})".format(s)

    def fill_memory(self):
        """Stores transitions produced by random actions in memory."""
        print("Filling memory...")
        while len(self.memory) < self.max_mem:
            x = self.env.reset()
            s = [x] * self.seq_length
            done = False
            while not done:
                a = self.env.action_space.sample()
                x, r, done, info = self.env.step(a)
                s_ = s.copy()
                s.pop(0)
                s.append(x)
                self.memory.append((np.array(s_.copy()),
                                    a,
                                    r,
                                    np.array(s.copy()),
                                    done))
                if len(self.memory) == self.max_mem:
                    print("Done.")
                    break

    def select_action(self, state):
        """
        Performs epsilon-greedy action selection.

        Returns:
            An integer between 0 and net.num_actions.

        """
        p = np.random.random()
        if p < self.epsilon:
            a = np.random.choice(self.net.actions)
        else:
            a = np.argmax(self.net.evaluate(np.array(state)))
        return a

    def targets(self, batch):
        # batch is [batch_size x 5]
        targets = []
        states_, actions, rewards, states, end_flags = batch
        values = self.net.evaluate(states, clone=True)
        for v, r, s, done in zip(values, rewards, states, end_flags):
            if done:
                targets.append(r)
            else:
                targets.append(r + self.gamma * np.max(v))
        return np.array(targets)

    def get_batch(self):
        batch = []
        idx = np.random.choice(range(0, self.max_mem), size=self.batch_size)
        for i in idx:
            batch.append(self.memory[i])
        batch = np.array(batch)
        s_ = np.array([s.ravel() for s in batch[:, 0]])
        a = np.zeros((self.batch_size, self.net.num_classes))
        a_idx = np.array([a for a in batch[:, 1]])
        a[[np.arange(0, self.batch_size), a_idx]] = 1
        r = np.array([r for r in batch[:, 2]])
        s = np.array([s.ravel() for s in batch[:, 3]])
        done = np.array([f for f in batch[:, 4]])
        return s_, a, r, s, done

    def run_episode(self):
        # Inner-loop of Algorithm 1.
        # NOTE: # paper uses s.extend([a, x]), but then doesn't use a... ?

        x = self.env.reset()
        s = [x] * self.seq_length
        rewards = []
        steps = 0
        done = False
        while not done:
            a = self.select_action(s)
            x, r, done, info = self.env.step(a)
            if r == 100:
                print("The ship has landed!")
            rewards.append(r)
            s_ = s.copy()
            s.pop(0)
            s.append(x)
            if len(self.memory) == self.max_mem:
                self.memory.pop(0)
            obs = (np.array(s_.copy()), a, r, np.array(s.copy()), done)
            self.memory.append(obs)
            s_, a, _, _, _ = batch = self.get_batch()
            y = self.targets(batch)
            self.net.update(s_, a, targets=y)
            steps += 1
            if steps % self.update_steps == 0:
                self.net.clone()
                print("Updated the clone network.")
        return self.score(rewards)

    def train(self):
        # Outer-loop of Algorithm 1.
        episodes = 0
        ave_score = 0
        self.fill_memory()
        while episodes < self.max_episodes:
            ave_score += self.run_episode()
            episodes += 1
            self.epsilon = (1.00 - self.min_epsilon) * self.decay_factor ** (episodes / self.decay_episodes)
            if episodes % 10 == 0:
                print("episode={:d}, ave_score={:.2f}, epsilon={:.2f}".format(episodes, ave_score / 10, self.epsilon))
                ave_score = 0
