import gymnasium as gym
from gymnasium.wrappers import TransformObservation
import math
import numpy as np
import os

import math_utils as mu
import learning.q_learning as qlearning
import policies.greedy as gp
import rl_utils as rlu

# Globals
ALPHA = 1                   # Learning rate, constant for first 1000 iterations
ALPHA_DECAY_BASE = 0.9997   # Starting value for alpha when decaying after the 1000th iteration
ALPHA_DECAY_START = 1000
GAMMA = 0.95                # Discount factor

# Initialize the environment
env = gym.make("CartPole-v1", render_mode="rgb_array")

# Transforming observation so that:
# Position: 2.3244 -> 2.25, steps of 0.25 [-0.25, 0, 0.25, 0.5, 0.75, ...]. Interval considered [-3.75, 3.75]
# Cart Velocity same as Position
# Pole Angle: 0.1254 -> 0.128 steps of 0.008 (0.5 deg_to_rad) [-0.008, 0, 0.008, 0.016, ...]. Interval considered [-0.24, 0.24]
# Pole angular velocity 0.46 -> 0.5 steps of 0.1 
state_space_shape = (30, 30, 50, 60)
state_space_scale = (0.25, 0.25, math.radians(0.5), 0.1)
env = TransformObservation(env, lambda obs: (mu.round_to(obs[0], state_space_scale[0], 2), mu.round_to(obs[1], state_space_scale[1], 2), mu.round_to(obs[2], state_space_scale[2], 3), mu.round_to(obs[3], state_space_scale[3], 1)))

# Initialize learning algorithm
learn = qlearning.QLearning(gamma=GAMMA, starting_alpha=ALPHA, decay_alpha=ALPHA_DECAY_BASE, start_decay_iter=ALPHA_DECAY_START)
learn.init_q_table(state_space_shape=state_space_shape, state_space_scale=state_space_scale, action_space=env.action_space)
learn.load(os.getcwd() + "/tables/cart_q_table.json")

# Initialize policy for testing
testing_policy = gp.GreedyPolicy(action_space = env.action_space)

rlu.Test(env, learn, testing_policy, 0, True)