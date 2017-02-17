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
        actions: `Tensor` of shape (batch_size, num_classes).
    """

    actions = tf.cast(actions, tf.float32)
    logits = tf.reduce_sum(tf.multiply(logits, actions), axis=1)
    errors = targets - logits
    # clipped = tf.maximum( tf.minimum(errors, 1), -1)
    mse = tf.reduce_mean( tf.square(errors) )
    # mse = tf.reduce_mean( tf.square(clipped) )
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
        self.graph = tf.Graph()
        if batch_size is not None:
            self.batch_size = batch_size
            if self.agent == 'reinforce':
                assert self.batch_size == 1, "REINFORCE requires that batch_size == 1."

    def __str__(self):
        return "Network()"

    def __repr__(self):
        return "<pf.Network>"

    def num_features(self):
        return self.layers[0]

    def num_classes(self):
        return self.layers[-1]

    def build(self):

        """Build the network. This must be called before the network is used."""

        with self.graph.as_default():

            # Data
            self.global_step = tf.Variable(0, trainable=False, name="global_step")
            self.features_pl = tf.placeholder(tf.float32, shape=(None, self.num_features()))
            self.actions_pl = tf.placeholder(tf.float32, shape=(None, self.num_classes()))
            self.prob_pl = tf.placeholder(tf.float32, shape=())
            if self.agent == 'reinforce':
                self.gain_pl = tf.placeholder(tf.float32, shape=())
            if self.agent == 'deepq':
                self.targets_pl = tf.placeholder(tf.float32, shape=(None, ))
            self.lr = tf.train.exponential_decay(self.learning_rate,
                                                 self.global_step,
                                                 self.decay_steps,
                                                 self.decay_factor)

            # Activation
            if self.type == 'fc':
                self.logits = fc_activation(self.layers, self.features_pl, self.prob_pl)
                self.softmax = tf.nn.softmax(self.logits)
                if self.agent == 'deepq':
                    self.targets = fc_activation(self.layers, self.features_pl, self.prob_pl, clone=True)
            elif self.type == 'cnn':
                pass
                # self.logits = cnn_activation(self.features_pl, self.prob_pl)
            elif self.type == 'rnn':
                pass
                # self.logits = rnn_activation(self.features_pl, self.prob_pl)
            else:
                print("ERROR: Unable to create a network. Network type {} type not recognized.".format(network['type']))

            # Training
            if self.agent == "reinforce":
                self.loss = reinforce_loss(self.logits, self.actions_pl, self.gain_pl)
                self.train_op = reinforce_update(self.loss,
                                                 self.learning_rate,
                                                 self.global_step,
                                                 self.decay_steps,
                                                 self.decay_factor)
            elif self.agent == "deepq":
                self.clone_weights = clone_weights(self.graph)
                self.loss = deepq_loss(self.logits, self.targets_pl, self.actions_pl)
                self.train_op = deepq_update(self.loss,
                                             self.learning_rate,
                                             self.global_step,
                                             self.decay_steps,
                                             self.decay_factor)
            elif self.agent == "trpo":
                pass
                # self.loss = trpo_loss(self.logits, self.actions_pl)
                # self.train_op = trpo_update(self.loss, self.global_step)
            else:
                print("ERROR: Unable to create a network. Agent type {} type not recognized.".format(agent))

            # Session
            self.init_op = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(self.init_op)
            self.clone() # start with same weights

    def close(self):
        self.sess.close()

    def evaluate(self, s, a=None, clone=False):
        """
        Network evaluated at s, then (optionally) at a.

        Args:
            s: a `Tensor` of shape (batch_size, num_features).
            a: an integer in the range (0, num_classes), or a list/array of integers in
               the same range.
        Returns:
            output: if a is None, an `Array` of size (batch_size, num_classes). Else,
                    an `Array` of size (batch_size, ).
        """

        feed_dict = {self.features_pl:s.reshape((-1, self.num_features())),
                     self.prob_pl:1.00}
        if clone:
            output = self.sess.run(self.targets, feed_dict=feed_dict)
        else:
            output = self.sess.run(self.logits, feed_dict=feed_dict)

        if a is None:
            return output
        else:
            shape = output.shape
            if len(shape) == 1:
                return output[a]
            else:
                return output[np.arange(0, shape[0]), a]

    def update(self, s, a, targets=None, gain=None):

        """
        Peforms a single training operation to update network parameters.

        Args:
            s: an array of shape (batch_size, num_features).
            a: an array of shape (batch_size, num_classes).
            targets: (for DeepQ) an array of shape (batch_size, ).
            gain: (for REINFORCE) a float.

        Returns:
            None.
        """

        if self.agent == 'reinforce':
            feed_dict = {self.features_pl:states.reshape(1, self.num_features()),
                         self.actions_pl:actions.reshape(1, self.num_classes),
                         self.gain_pl:gain,
                         self.prob_pl:self.keep_prob}
            self.sess.run(self.train_op, feed_dict=feed_dict)
        elif self.agent == 'deepq':
            s = s.reshape(-1, self.num_features())
            a = a.reshape(-1, self.num_classes())
            assert s.shape[0] == a.shape[0], "Number of states and actions don't match."
            assert s.shape[0] == targets.shape[0], "Number of states and targets don't match."
            feed_dict = {self.features_pl:s,
                         self.actions_pl:a,
                         self.targets_pl:targets,
                         self.prob_pl:self.keep_prob}
            loss, _ = self.sess.run([self.loss, self.train_op], feed_dict=feed_dict)
            return loss

    def choice(self, s):
        """Sample an action from probabilities given by forward pass of s."""
        assert len(s.shape) == 1, "choice expected 1-dimensional input"
        feed_dict = {self.features_pl:s,
                     self.prob_pl:1.00}
        p = self.sess.run(self.softmax, feed_dict=feed_dict)
        return np.random.choice(self.actions, p=p.reshape(len(self.actions),))

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

    """
        Implements a DQN learner as described in [].

        In the value of epsilon in epsilon-greedy
        action selection is according to the schedule:

            epsilon(t) = (max - min) * decay_factor ** (t / decay_episodes),

        where t is measured in episodes.

    """

    def __init__(self, net, env, discount_rate, max_episodes, score, min_mem,
                 max_mem, seq_length, batch_size, print_epsidoes, decay_episodes,
                 decay_factor, clone_steps, init_epsilon, min_epsilon):
        Agent.__init__(self, net, env, discount_rate, max_episodes, score)
        self.epsilon = init_epsilon
        self.init_epsilon = init_epsilon
        self.min_epsilon = min_epsilon
        self.memory = []
        self.min_mem = min_mem
        self.max_mem = max_mem
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.print_epsidoes = print_epsidoes
        self.decay_episodes = decay_episodes
        self.decay_factor = decay_factor
        self.clone_steps = clone_steps

    def __str__(self):
        s = format(self.env).rstrip(" instance>") + ">"
        return "Deep-Q agent for environment {}.".format(s)

    def __repr__(self):
        s = format(self.env).rstrip(" instance>") + ">"
        return "Deep-Q agent(environment={})".format(s)

    def init_memory(self):
        """Stores transitions produced by random actions in memory."""
        print("Initializing memory / exploration...")
        while len(self.memory) < self.min_mem:
            x = self.env.reset()
            s = [x] * self.seq_length
            done = False
            while not done:
                a = self.env.action_space.sample()
                x, r, done, info = self.env.step(a)
                s_ = s.copy()
                s.pop(0)
                s.append(x)
                self.memory.append((np.array(s_.copy()).reshape((1,-1)),
                                    a,
                                    r,
                                    np.array(s.copy()).reshape((1,-1)),
                                    done))
                if len(self.memory) == self.min_mem:
                    print("Done.")
                    break

    def select_action(self, s):
        """
        Performs epsilon-greedy action selection.

        Args:
            s: array with shape (1, num_features).

        Returns:
            An integer between 0 and net.num_actions.

        """

        assert len(s.shape) == 1, "Greedy action selection requires len(state) == 1."

        p = np.random.random()
        if p < self.epsilon:
            a = np.random.choice(self.net.actions)
        else:
            a = np.argmax(self.net.evaluate(np.array(s)))
        return a

    def targets(self, states, rewards, end_flags):

        """
            Compute the targets for Deep-Q updates.

            Args:
                r: array of shape (batch_size, )
                s: array of shape (batch_size, num_features)
                end_flags: array of shape (batch_size, ). Boolean values
                           indicate the end of an episode was reached.

            Returns:
                targets: an array of shape (batch_size, ).
        """

        values = self.net.evaluate(states, clone=True)
        targets = rewards + self.discount_rate * np.max(values, axis=1)
        targets[end_flags==True] = rewards[end_flags==True]
        return targets

    def sample_memory(self, n):
        idx = np.random.choice(range(0, len(self.memory)), size=n)
        sample = np.array(self.memory)[idx, :]
        return sample

    def get_batch(self, mem):
        idx = np.random.choice(range(0, len(mem)), size=self.batch_size)
        batch = mem[idx,:]
        s_ = np.concatenate(batch[:,0], axis=0)
        a_idx = batch[:,1]
        a = np.zeros((self.batch_size, self.net.num_classes()))
        a[[np.arange(0, self.batch_size), a_idx.astype('int')]] = 1
        r = batch[:,2]
        s = np.concatenate(batch[:,3], axis=0)
        done = batch[:,4]
        return s_, a, r, s, done

    def run_episode(self, total_steps=0):
        # Inner-loop of Algorithm 1.
        # NOTE: # paper uses s.extend([a, x]), but then doesn't use a... ?

        x = self.env.reset()
        s = [x] * self.seq_length
        r_list = []
        loss_list = []
        done = False
        sample = self.sample_memory(self.batch_size * 100)
        while not done:
            a = self.select_action(np.array(s).reshape((-1)))
            x, r, done, info = self.env.step(a)
            # print("took a step")
            r_list.append(r)
            s_ = s.copy()
            s.pop(0)
            s.append(x)
            if len(self.memory) == self.max_mem:
                self.memory.pop(0)
            self.memory.append((np.array(s_.copy()).reshape((1,-1)),
                                a,
                                r,
                                np.array(s.copy()).reshape((1,-1)),
                                done))
            # print("updated the memory bank")
            states_, actions, rewards, states, end_flags = self.get_batch(sample)
            # print("grabbed a batch of memories")
            targets = self.targets(states, rewards, end_flags)
            loss = self.net.update(states_, actions, targets=targets)
            loss_list.append(loss)
            # print("performed an update")
            total_steps += 1
            if total_steps % self.clone_steps == 0:
                self.net.clone()
                print("Updated the clone network (step={}).".format(total_steps))
            if done:
                if r == 100:
                    print("The ship has landed!")
                else:
                    # print("The ship crashed...")
                    pass
        return np.mean(loss_list), self.score(r_list), total_steps

    def train(self):
        # Outer-loop of Algorithm 1.

        episodes = 0
        total_steps = 0
        ave_score = 0  # over last print_epsidoes episodes
        ave_loss = 0  # over last print_epsidoes episodes
        self.init_memory()
        while episodes < self.max_episodes:
            loss, score, total_steps = self.run_episode(total_steps)
            ave_loss += loss
            ave_score += score
            episodes += 1
            self.epsilon = (self.init_epsilon - self.min_epsilon) * self.decay_factor ** (episodes / self.decay_episodes) + self.min_epsilon
            if episodes % self.print_epsidoes == 0:
                lr = self.net.sess.run(self.net.lr)
                print("episode={:d}, lr={:.8f}, total_steps={:d}, ave_loss={:.2f}, ave_score={:.2f}, epsilon={:.2f}".format(episodes, lr, total_steps, ave_loss / self.print_epsidoes, ave_score / self.print_epsidoes, self.epsilon))
                ave_score = 0
                ave_loss = 0
