import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import core as pf
import gym

def run_episode(env, render=False, learner=None):
    s = env.reset()
    steps = 0
    done = False
    while not done:
        if render:
            env.render()
        if learner:
            action = learner.net.choice(s.reshape(1,4))
        else:
            action = env.action_space.sample()
        s, r, done, info = env.step(action)
        steps += 1
    return steps

def score(rewards):
    """Score for the cart pole environment."""
    return len(rewards)

# KEEP_PROB = 1.00
# ACTIONS = [0,1]
# DISCOUNT_RATE = 1.00
#
# net = pf.Network(actions=ACTIONS, keep_prob=KEEP_PROB)
#
# env = gym.make('CartPole-v0')
#
# learner = pf.Reinforce(net=net, env=env, discount_rate=DISCOUNT_RATE, score=score)
#
# learner.train()
#
# episodes = 100
# reinforce = 0
# random = 0
# for _ in range(episodes):
#     reinforce += run_episode(env, render=False, learner=learner)
#     random += run_episode(env, render=False)
# print("REINFORCE average = {}".format(reinforce / episodes))
# print("Random average = {}".format(random / episodes))
# plt.plot(learner.scores)
# plt.show()


# 2/12/17
nn = 512
num_features = 4
num_actions = 2
max_episodes = 200
learning_rate = 0.0001
decay_steps = 1000
decay_factor = 0.90
network = {'type': 'fc', 'layers': [4, 512, 256, 2]}
actions = [0, 1]
discount_rate = 1.00

net = pf.Network(network, 'reinforce', actions, learning_rate, decay_steps, decay_factor)
env = gym.make('CartPole-v0')
learner = pf.Reinforce(net=net,
                       env=env,
                       discount_rate=discount_rate,
                       max_episodes=max_episodes,
                       score=score)

learner.train()

episodes = 100
reinforce = 0
random = 0
for _ in range(episodes):
    reinforce += run_episode(env, render=False, learner=learner)
    random += run_episode(env, render=False)
print("REINFORCE average = {}".format(reinforce / episodes))
print("Random average = {}".format(random / episodes))
plt.plot(learner.scores)
plt.show()
