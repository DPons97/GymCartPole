import gymnasium as gym
from gymnasium.wrappers import TransformObservation
import math
import numpy as np

import math_utils as mu

# Initialize the environment
env = gym.make("CartPole-v1", render_mode = "human")

# Transforming observation so that:
# Position: 2.3344 -> 2.3, steps of 0.1 [-0.1, 0, 0.1, 0.2, 0.3, ...]
# Cart Velocity ignored for now
# Pole Angle: 0.1254 -> 0.1221 steps of (1 deg_to_rad) [-0.017, 0, 0.017, 0.034, ...]
# Pole angular Velocity ignored for now
env = TransformObservation(env, lambda obs: (mu.round_to(obs[0], 0.1), mu.round_to(obs[1], 0.1, 1), mu.round_to(obs[2], math.radians(1), 4), mu.round_to(obs[3], 0.1, 1)))

prev_obs, info = env.reset()
action = env.action_space.sample()
for _ in range(1000):
    curr_obs, reward, term, trunc, info = env.step(action)
    print("New observation: " + str(curr_obs))
    print("Reward: " + str(reward))
    
    if (term):
        prev_obs, info = env.reset()
        print("Episode terminated!")
    else:
        prev_obs = curr_obs
    
    action = env.action_space.sample()
    print("Next action will be: " + ("L" if action==0 else "R"))

quit()