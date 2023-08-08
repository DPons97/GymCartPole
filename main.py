import gymnasium as gym
from gymnasium.wrappers import TransformObservation
import math
import random as rand
import numpy as np

import math_utils as mu
import scaled_ndarray as sa

# Globals
N_ITERATIONS = 2000     # Number of iterations (simulations)
ALPHA = 0.1             # Learning rate
GAMMA = 0.75            # Discount factor
EPS = 0.995             # Epsilon-greedy starting randomness

'''
    Update the Q_value(s, a) related to the last executed action starting from the previous state
'''
def update_q_value(q_table, prev_state, action, curr_state, reward, alpha, gamma):    
    prev_q = q_table[prev_state + (action,)]        # Q-value to be updated
    curr_q = q_table[curr_state]                    # (Q_value[L], Q_value[R])

    prev_q = (1-alpha)*prev_q + alpha*(reward + gamma*(max(curr_q))) 
    q_table[prev_state + (action,)] = prev_q
    return prev_q

'''
    Choose next action to be executed, with epsilon-greedy policy. 
    q_table - Table with Q-values of current learning
    curr_state - Current observation from which to decide the next action
    epsilon - Probability to make a random move instead of the optimal action (i.e. probability to explore instead of exploit)
'''
def next_action_epsilon_greedy(q_table, curr_state, epsilon):
    if rand.randint(1, 100) <= epsilon*100:
        return env.action_space.sample()
    else:
        curr_q = q_table[curr_state]                    # (Q_value[L], Q_value[R])
        return np.argmax(curr_q)

# Initialize the environment
env = gym.make("CartPole-v1", render_mode = "human")

# Transforming observation so that:
# Position: 2.3344 -> 2.3, steps of 0.1 [-0.1, 0, 0.1, 0.2, 0.3, ...]
# Cart Velocity ignored for now
# Pole Angle: 0.1254 -> 0.1221 steps of (1 deg_to_rad) [-0.017, 0, 0.017, 0.034, ...]
# Pole angular Velocity ignored for now
env = TransformObservation(env, lambda obs: (mu.round_to(obs[0], 0.1), mu.round_to(obs[2], math.radians(1), 4)))

# Policy and Q-table 
state_space_shape = (94, 48, 2)
state_space_scale = (0.1, math.radians(1), 1)
# cart_policy = sa.ScaledNDArray(state_space_shape, state_space_scale)
q_table = sa.ScaledNDArray(state_space_shape, state_space_scale)
best_score = (0, EPS)   # (score, randomness)

prev_obs, info = env.reset()
action = env.action_space.sample()
curr_score = 0
for _ in range(N_ITERATIONS):
    curr_obs, reward, term, trunc, info = env.step(action)
    curr_score += reward
    print("New observation: " + str(curr_obs))
    print("Reward: " + str(reward))

    #if (term):
        # Update policy with negative reward
    #    reward = -1

    new_q_value = update_q_value(q_table=q_table, prev_state=prev_obs, action=action, curr_state=curr_obs, reward=reward, alpha=ALPHA, gamma=GAMMA)
    print("New q value for " + str(prev_obs + (action,)) + " = " + str(new_q_value))
    
    if (term):
        # Store best score if gained more points or if point are equal but randomness is lower
        if (curr_score >= best_score[0]):
            best_score = (curr_score, EPS)

        curr_score = 0
        prev_obs, info = env.reset()
        EPS *= EPS
        print("Episode terminated!")
        print("New probability to make a random move: " + str(EPS))
    else:
        prev_obs = curr_obs
    
    action = next_action_epsilon_greedy(q_table, curr_obs, EPS)
    print("Next action will be: " + ("L" if action==0 else "R"))

print("Best score: " + str(best_score))
quit()