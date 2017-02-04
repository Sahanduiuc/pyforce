import tensorflow as tf
import numpy as np
import core as pf
import gym

nn = 64
k = 4
action_space = [0,1]
discount = 0.99
alpha = 0.0001
max_episodes = 10000

def shape(tensor):
    """Returns the shape of the tensor as a tuple. A more useful alternative to
       tf.get_shape()."""
    shape = tensor.get_shape()
    return tuple([shape[i].value for i in range(0, len(shape))])

def placeholders(batch_size, num_features):
    features_pl = tf.placeholder(tf.float32, shape=[batch_size, num_features])
    labels_pl = tf.placeholder(tf.int32, shape=[batch_size, ])
    return features_pl, labels_pl

def feed(data, features_pl, labels_pl, batch_size):
    """Feed the next batch of data to placeholders.

    Arguments
    ---------
    data (dict): datasets to get batches from.

    Returns
    -------
    A dictionary of placeholders (key) and associated data (value).

    """

    features = data['features'].get_batch(batch_size, demean=True)
    labels = data['labels'].get_batch(batch_size)
    feed_dict = {features_pl: features, labels_pl: labels}
    return feed_dict

def activation(features_pl):
    k, = shape(features_pl)
    x = tf.reshape(state_pl, shape=(1,k))
    with tf.name_scope('full'):
        sd = 1.0 / np.sqrt(float(k))
        weights = tf.Variable(tf.truncated_normal([k, nn], stddev=sd), name='weights')
        bias = tf.Variable(tf.constant(0.0, shape=[nn]), name='bias')
        full =  tf.matmul(x, weights) + bias
    with tf.name_scope('softmax'):
        sd = 1.0 / np.sqrt(float(nn))
        weights = tf.Variable(tf.truncated_normal([nn, 2], stddev=sd), name='weights')
        bias = tf.Variable(tf.constant(0.0, shape=[2]), name='bias')
        logits =  tf.matmul(full, weights) + bias
        pi = tf.nn.softmax(logits)
    return pi

def gradient(y):
    _, k = shape(y)
    G = [tf.gradients(indicator(y, i), weights) for i in np.arange(0, k)]
    return G

def updates(weights, gradients, value, gain, alpha):
    """This will be added to the graph at the same time as the weight and gradient
       tensors. Also need to add gain, value, and alpha placeholders to the graph.
       Then this operation can be called by running sess.run(update, feed_dict=...).
    """

    ops = []
    for Df in gradients:
        sub_ops = []
        for w,df in zip(weights, Df):
            # print("added an update op")
            sub_ops.append(tf.assign(w, tf.add(w, tf.multiply(alpha, tf.multiply(gain, tf.divide(df, value))))))
        ops.append(sub_ops)
    return ops

def indicator(tensor, index):
    _, k = tensor.get_shape().as_list()  # assume it is (num_examples x k)
    s = np.zeros([1, k])
    s[0, index] = 1
    return tensor * s

def softmax(logits):
    return np.exp(logits) / np.sum(np.exp(logits))

def run_episode(env, render=False, learner=None):
    s = env.reset()
    steps = 0
    done = False
    while not done:
        if render:
            env.render()
        if learner:
            action = learner.network.choice(s)
        else:
            action = env.action_space.sample()
        s, r, done, info = env.step(action)
        steps += 1
    return steps

# Initialize the network
with tf.Graph().as_default():
    state_pl = tf.placeholder(tf.float32, shape=(k,))
    gain_pl = tf.placeholder(tf.float32, shape=())
    alpha_pl = tf.placeholder(tf.float32, shape=())
    value_pl = tf.placeholder(tf.float32, shape=())
    placeholders = {'state_pl': state_pl,
                    'gain_pl': gain_pl,
                    'alpha_pl': alpha_pl,
                    'value_pl': value_pl}
    eval_op = activation(state_pl)
    weights = tf.get_collection('trainable_variables')  # must come AFTER eval_op!
    diff_op = gradient(eval_op)
    update_op = updates(weights, diff_op, value_pl, gain_pl, alpha_pl)
    init = tf.global_variables_initializer()
    sess = tf.Session()
network = pf.Network(eval_op, weights, action_space, diff_op, update_op, sess, placeholders, init)
network.initialize()

# Initialize the environment
env = gym.make('CartPole-v0')

# Initialize the learner
learner = pf.Reinforce(network=network, env=env, max_episodes=max_episodes, alpha=alpha)

# Train the learner
learner.train()

# Test the learner
episodes = 1000
reinforce = 0
random = 0
for _ in range(episodes):
    reinforce += run_episode(env, render=False, learner=learner)
    random += run_episode(env, render=False)
print("REINFORCE average = {}".format(reinforce / episodes))
print("Random average = {}".format(random / episodes))
