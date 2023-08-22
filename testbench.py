import gymnasium as gym
from gymnasium.wrappers import TransformObservation
import math
import numpy as np
import os

import learning.q_learning as qlearning

import math_utils as mu

# Initialize the environment
env = gym.make("CartPole-v1", render_mode = "human")

# Transforming observation so that:
# Position: 2.3244 -> 2.25, steps of 0.25 [-0.25, 0, 0.25, 0.5, 0.75, ...]. Interval considered [-3.75, 3.75]
# Cart Velocity same as Position
# Pole Angle: 0.1254 -> 0.128 steps of 0.008 (0.5 deg_to_rad) [-0.008, 0, 0.008, 0.016, ...]. Interval considered [-0.24, 0.24]
# Pole angular velocity 0.46 -> 0.5 steps of 0.1 
state_space_shape = (30, 30, 50, 60)
state_space_scale = (0.25, 0.25, math.radians(0.5), 0.1)
env = TransformObservation(env, lambda obs: (mu.round_to(obs[0], state_space_scale[0], 2), mu.round_to(obs[1], state_space_scale[1], 2), mu.round_to(obs[2], state_space_scale[2], 3), mu.round_to(obs[3], state_space_scale[3], 1)))

learn = qlearning.QLearning(gamma=0, starting_alpha=0, decay_alpha=0, start_decay_iter=1)
learn.init_q_table(state_space_shape=state_space_shape, state_space_scale=state_space_scale, action_space=env.action_space)
learn.load(os.getcwd() + "/tables/cart_q_table.json")

obs, info = env.reset()
term = False
total_score = 0
while not term:
    curr_q = learn._q_table[obs]                    # (Q_value[a_1], Q_value[a_2], ...)
    action = np.argmax(curr_q)
    obs, reward, term, trunc, info = env.step(action)

    total_score += reward
    
print("Total score: " + str(total_score))