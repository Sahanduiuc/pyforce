import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import core as pf
import gym
import sys

_, example, agent = sys.argv

def run_episode(env, render=False, agent=None, score=None):
    s = env.reset()
    rewards = []
    done = False
    while not done:
        if render:
            env.render()
        if agent:
            action = agent.net.choice(s.reshape(1, agent.net.num_features))
        else:
            action = env.action_space.sample()
        s, r, done, info = env.step(action)
        rewards.append(r)
    return score(rewards)

def evaluate(env, agent, score, episodes):
    print("Evaluating the agent...")
    agent_score = 0
    random_score = 0
    for step in range(episodes):
        agent_score += run_episode(env, render=False, agent=agent, score=score)
        random_score += run_episode(env, render=False, score=score)
        # print("step={}".format(step + 1))
    print("Agent average score={}".format(agent_score / episodes))
    print("Random average score={}".format(random_score / episodes))
    plt.plot(agent.scores)
    plt.show()
    plt.clf()

# CartPole-v0
if example == "CartPole":
    def score(rewards):
        return np.sum(rewards)
    env = gym.make('CartPole-v0')
    net = pf.Network(network={'type': 'fc', 'layers': [4, 64, 64, 2]},
                     agent='reinforce',
                     actions=[0, 1],
                     learning_rate=0.0001,
                     decay_steps=1000,
                     decay_factor=0.90,
                     keep_prob=1.00)
    agent = pf.Reinforce(net=net,
                           env=env,
                           discount_rate=1.00,
                           max_episodes=500,
                           score=score)
    agent.train()
    evaluate(env, agent, score, 100)

# LunarLander-v1
if example == "LunarLander":
    def score(rewards):
        return np.sum(rewards)
    env = gym.make('LunarLander-v2')
    net = pf.Network(network={'type': 'fc', 'layers': [8, 64, 64, 4]},
                     agent='reinforce',
                     actions=[0, 1, 2, 3],
                     learning_rate=0.0001,
                     decay_steps=1000,
                     decay_factor=0.90,
                     keep_prob=1.00)
    agent = pf.Reinforce(net=net,
                           env=env,
                           discount_rate=0.90,
                           max_episodes=100,
                           score=score)
    agent.train()
    evaluate(env, agent, score, 100)

# Pong-v0
if example == "Pong":
    def score(rewards):
        return np.mean(rewards)
    env = gym.make('Pong-v0')
    net = pf.Network(network={'type': 'cnn', 'layers': [(210, 160, 3), 512, 256, 2]},
                     agent='reinforce',
                     actions=[0, 1, 2, 3, 4, 5],
                     learning_rate=0.0001,
                     decay_steps=1000,
                     decay_factor=0.90)
    agent = pf.Reinforce(net=net,
                           env=env,
                           discount_rate=1.00,
                           max_episodes=200,
                           score=score)
    agent.train()
    evaluate(env, agent, 100)

# LunarLander-v2 (DeepQ)
if example == "LunarLander" and agent == "DeepQ":
    def score(rewards):
        return np.sum(rewards)
    env = gym.make('LunarLander-v2')
    net = pf.Network(network={'type': 'fc', 'layers': [8, 64, 64, 4]},
                     agent='deepq',
                     actions=[0, 1, 2, 3],
                     learning_rate=0.0001,
                     decay_steps=100,
                     decay_factor=0.99,
                     keep_prob=1.00)
    agent = pf.Reinforce(net=net,
                           env=env,
                           discount_rate=0.90,
                           max_episodes=500,
                           score=score)
    agent.train()
    evaluate(env, agent, score, 100)
