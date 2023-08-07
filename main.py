import gymnasium as gym
from gymnasium.wrappers import TransformObservation
import math
import random as rand
import numpy as np

import math_utils as mu
import RL.GymCartPole.policy as sp

env = gym.make("CartPole-v1", render_mode = "human")

# Transforming observation so that:
# Position: 2.3344 -> 2.3, steps of 0.1 [-0.1, 0, 0.1, 0.2, 0.3, ...]
# Cart Velocity ignored for now
# Pole Angle: 0.1254 -> 0.1221 steps of (1 deg_to_rad) [-0.017, 0, 0.017, 0.034, ...]
# Pole angular Velocity ignored for now
env = TransformObservation(env, lambda obs: (mu.round_to(obs[0], 0.1), mu.round_to(obs[2], math.radians(1), 4)))

# 0 - left
# 1 - right
action_space = [0, 1]

state_space = sp.Policy((94, 48, 2), (0.1, math.radians(1), 1))
observation, info = env.reset()
action = rand.randint(0, 1)
for _ in range(1000):
    observation, reward, term, trunc, info = env.step(action)

    print(observation)

    if (term):
        env.reset()
        quit()