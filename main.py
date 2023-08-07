import gymnasium as gym
from gymnasium.wrappers import TransformObservation
import math
import random as rand
import numpy as np

import math_utils as mu
import scaled_ndarray as sa

def update_q_value(q_table, prev_state, action, curr_state, reward, alpha, gamma):    
    prev_q = q_table[prev_state + (action,)]        # Q-value to be updated
    curr_q = q_table[curr_state]                    # (Q_value[L], Q_value[R])

    prev_q = (1-alpha)*prev_q + alpha*(reward + gamma*(max(curr_q))) 
    q_table[prev_state + (action,)] = prev_q
    return prev_q

def choose_next_action(q_table, curr_state):
    curr_q = q_table[curr_state]                    # (Q_value[L], Q_value[R])
    return np.argmax(curr_q)


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

alpha = 0.5     # Learning rate
gamma = 0.5       # Discount factor

# Policy and Q-table 
state_space_shape = (94, 48, 2)
state_space_scale = (0.1, math.radians(1), 1)
cart_policy = sa.ScaledNDArray(state_space_shape, state_space_scale)
q_table = sa.ScaledNDArray(state_space_shape, state_space_scale)

prev_obs, info = env.reset()
action = env.action_space.sample()
for _ in range(1000):
    curr_obs, reward, term, trunc, info = env.step(action)
    print("New observation: " + str(curr_obs))
    print("Reward: " + str(reward))

    new_q_value = update_q_value(q_table=q_table, prev_state=prev_obs, action=action, curr_state=curr_obs, reward=reward, alpha=alpha, gamma=gamma)
    print("New q value for " + str(prev_obs + (action,)) + " = " + str(new_q_value))

    prev_obs = curr_obs
    action = choose_next_action(q_table, curr_obs)
    print("Next action will be: " + "L" if 0 else "R")
    
    if (term):
        # todo Update policy with negative reward?
        env.reset()

quit()