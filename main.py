import gymnasium as gym
from gymnasium.wrappers import TransformObservation
import math
import matplotlib.pyplot as plt
import numpy as np

import math_utils as mu
import learning.q_learning as qlearning
import policies.epsilon_greedy as ep

# Globals
N_ITERATIONS = 1000         # Number of iterations (simulations)
MAX_TIMESTEPS = 1000        # Maximum number of timesteps to be simulated during a single iteration
ALPHA = 0.1                 # Learning rate
GAMMA = 0.6                 # Discount factor
EPS = 0.995                  # Epsilon-greedy starting randomness

# Initialize the environment
env = gym.make("CartPole-v0")#, render_mode = "human")

# Transforming observation so that:
# Position: 2.3344 -> 2.3, steps of 0.1 [-0.1, 0, 0.1, 0.2, 0.3, ...]
# Cart Velocity ignored for now
# Pole Angle: 0.1254 -> 0.1221 steps of (1 deg_to_rad) [-0.017, 0, 0.017, 0.034, ...]
# Pole angular Velocity ignored for now
state_space_shape = (94, 50, 48, 100)
state_space_scale = (0.1, 0.1, math.radians(1), 0.1)
env = TransformObservation(env, lambda obs: (mu.round_to(obs[0], state_space_scale[0]), mu.round_to(obs[1], state_space_scale[1], 1), mu.round_to(obs[2], state_space_scale[2], 4), mu.round_to(obs[3], state_space_scale[3], 1)))

# Initialize learning algorithm
learn = qlearning.QLearning(alpha=ALPHA, gamma=GAMMA)
learn.InitQTable(state_space_shape=state_space_shape, state_space_scale=state_space_scale, action_space=env.action_space)

# Initialize policy
policy = ep.EpsilonGreedyPolicy(action_space = env.action_space, eps = EPS)

# Init score data
scores = np.zeros(N_ITERATIONS)

# First action of first iteration is always random
action = policy.random_action()
prev_obs, info = env.reset()
for iter in range(1, N_ITERATIONS+1):
    print("Starting " + str(iter) + "th iteration...")
    curr_score = 0

    for t in range(MAX_TIMESTEPS):
        curr_obs, reward, term, trunc, info = env.step(action)
        print("New observation: " + str(curr_obs))
        curr_score += reward
        
        new_q_value = learn.update_q_value(prev_state=prev_obs, action=action, curr_state=curr_obs, reward=reward)
        print("New q value for " + str(prev_obs + (action,)) + " = " + str(new_q_value))
        
        if (term):
            break
        
        prev_obs = curr_obs
        action = policy.next_action(learn._q_table, curr_obs)
        print("Next action will be: " + ("L" if action==0 else "R"))
    
    # Iteration ended: Store the score, update hyperparameters
    scores[iter-1] = curr_score
    policy.set_epsilon(policy.epsilon() * EPS)
    learn.decay_learning_rate(iter)
    print("New probability to make a random move: " + str(policy.epsilon()))

    print("Episode terminated!")
    
    # Reset environment and choose first action of next iteration
    prev_obs, info = env.reset()
    action = policy.next_action(learn._q_table, curr_obs)

# Show stats
fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Cumulative Reward')
ax1.plot(range(0, N_ITERATIONS), scores, color)
ax1.tick_params(axis='y')

ax2 = ax1.twinx()

color = 'tab:red'
ax2.set_ylabel('Randomness (%)')
ax2.plot(range(0, N_ITERATIONS), [pow(EPS, i)  for i in range(0, N_ITERATIONS)], color=color)
ax2.tick_params(axis='y')

fig.tight_layout()
plt.show()

quit()