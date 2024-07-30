import gymnasium as gym
from gymnasium.wrappers import TransformObservation
import math
import numpy as np
import os

import learning.q_learning as qlearning
import policies.greedy as greedy

import math_utils as mu

# Initialize the environment
env = gym.make("CartPole-v1", render_mode = "human")

# Transforming observation so that:
# Position: 2.3244 -> 2.25, steps of 0.25 [-0.25, 0, 0.25, 0.5, 0.75, ...]. Interval considered [-3.75, 3.75]
# Cart Velocity same as Position
# Pole Angle: 0.1254 -> 0.128 steps of 0.008 (0.5 deg_to_rad) [-0.008, 0, 0.008, 0.016, ...]. Interval considered [-0.24, 0.24]
# Pole angular velocity 0.46 -> 0.5 steps of 0.1 
state_space_shape = (11, 11, 11, 11)
pos_space = np.linspace(-2.4, 2.4, 10)
vel_space = np.linspace(-4, 4, 10)
ang_space = np.linspace(-.2095, .2095, 10)
ang_vel_space = np.linspace(-4, 4, 10)

env = TransformObservation(env, lambda obs: (np.digitize(obs[0], pos_space), np.digitize(obs[1], vel_space), np.digitize(obs[2], ang_space), np.digitize(obs[3], ang_vel_space)))

learn = qlearning.QLearning(gamma=0, starting_alpha=0)
learn.init_q_table(state_space_shape=state_space_shape, action_space=env.action_space)

policy = greedy.GreedyPolicy(action_space=env.action_space)

learn.load(os.getcwd() + "/tables/cart_q_table.json")

obs, info = env.reset()
term = False
total_score = 0
while not term:
    action = policy.next_action(learn._q_table, obs)
    obs, reward, term, trunc, info = env.step(action)

    total_score += reward
    
print("Total score: " + str(total_score))