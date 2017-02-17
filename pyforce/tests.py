import tensorflow as tf
import numpy as np
import core as pf
import time
import gym
import sys

def timeit(func, reps=1, args=None):
    start = time.time()
    for _ in range(0, reps):
        func()
    stop = time.time()
    print("elapsed time: {:.4f}".format(stop - start))

batch_size = 32
seq_length = 1

net = pf.Network(network={'type': 'fc', 'layers': [4 * seq_length, 256, 256, 2]},
                 agent='deepq',
                 actions=[0, 1],
                 learning_rate=0.00025,  # 0.00025
                 decay_steps=1000,
                 decay_factor=0.99,
                 keep_prob=1.00,
                 batch_size=batch_size)
net.build()

env = gym.make('CartPole-v0')
def score(rewards):
    return np.sum(rewards)
agent = pf.DeepQ(net=net,
                 env=env,
                 discount_rate=0.99,  # 0.99
                 max_episodes=10000,
                 score=score,
                 min_mem=10 ** 4,  # set to 5% of max_mem
                 max_mem=10 ** 5,  # 10 ** 6
                 seq_length=seq_length,
                 batch_size=batch_size,
                 print_epsidoes=10,
                 decay_episodes=4,  # set so that exploration in 1/50 of total
                 decay_factor=0.975,  # set so that exploration in 1/50 of total
                 clone_steps=1000,  # 10 ** 4
                 init_epsilon=1.00,  # 1.00
                 min_epsilon=0.10)  # 0.10
agent.train()
#
# def run_episode(env, render=False, agent=None, score=None):
#     s = env.reset()
#     rewards = []
#     done = False
#     while not done:
#         if render:
#             env.render()
#         if agent:
#             action = agent.net.choice(s.reshape(1, agent.net.num_features()))
#         else:
#             action = env.action_space.sample()
#         s, r, done, info = env.step(action)
#         rewards.append(r)
#     if score is not None:
#         return score(rewards)
# run_episode(env, render=True)


# def run_episode(agent, total_steps=0):
#     # Inner-loop of Algorithm 1.
#     # NOTE: # paper uses s.extend([a, x]), but then doesn't use a... ?
#
#     x = agent.env.reset()
#     s = [x] * agent.seq_length
#     r_list = []
#     loss_list = []
#     done = False
#     sample = agent.sample_memory(agent.batch_size * 100)
#     while not done:
#         a = agent.select_action(np.array(s).reshape((-1)))
#         x, r, done, info = agent.env.step(a)
#         # print("took a step")
#         r_list.append(r)
#         s_ = s.copy()
#         s.pop(0)
#         s.append(x)
#         if len(agent.memory) == agent.max_mem:
#             agent.memory.pop(0)
#         agent.memory.append((np.array(s_.copy()).reshape((1,-1)),
#                             a,
#                             r,
#                             np.array(s.copy()).reshape((1,-1)),
#                             done))
#         # print("updated the memory bank")
#         states_, actions, rewards, states, end_flags = agent.get_batch(sample)
#         # print("grabbed a batch of memories")
#         targets = agent.targets(states, rewards, end_flags)
#         loss = agent.net.update(states_, actions, targets=targets)
#         loss_list.append(loss)
#         # print("performed an update")
#         total_steps += 1
#         if total_steps % agent.clone_steps == 0:
#             agent.net.clone()
#             # print("Updated the clone network (step={}).".format(total_steps))
#         if done:
#             if r == 100:
#                 print("The ship has landed!")
#             else:
#                 # print("The ship crashed...")
#                 pass
#     return np.mean(loss_list), agent.score(r_list), total_steps
