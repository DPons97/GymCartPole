import gymnasium as gym
from gymnasium.wrappers import TransformObservation
import math
import matplotlib.pyplot as plt
import numpy as np
import cv2

import math_utils as mu
import learning.q_learning as qlearning
import policies.epsilon_greedy as ep

# Globals
N_ITERATIONS = 4000         # Number of iterations (simulations)
MAX_TIMESTEPS = 1000        # Maximum number of timesteps to be simulated during a single iteration
ALPHA = 1                   # Learning rate, constant for first 1000 iterations
ALPHA_DECAY_BASE = 0.997    # Starting value for alpha when decaying after the 1000th iteration
GAMMA = 0.95                # Discount factor
EPS = 1                     # Epsilon-greedy starting randomness, constant for first 1000 iterations
EPS_DECAY_BASE = 0.997      # Starting value for epsilon when decaying after the 1000th iteration

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
learn = qlearning.QLearning(gamma=GAMMA, starting_alpha=ALPHA, decay_alpha=ALPHA_DECAY_BASE, start_decay_iter=500)
learn.InitQTable(state_space_shape=state_space_shape, state_space_scale=state_space_scale, action_space=env.action_space)

# Initialize policy
policy = ep.EpsilonGreedyPolicy(action_space = env.action_space, starting_eps = EPS, decay_eps = EPS_DECAY_BASE, start_decay_iter=500)

# Init stats data
scores = np.zeros(N_ITERATIONS)
epsilons = np.ones(N_ITERATIONS)

# First action of first iteration is always random
action = policy.random_action()
prev_obs, info = env.reset()
for iter in range(1, N_ITERATIONS+1):
    print("Starting " + str(iter) + "th iteration...")
    curr_score = 0

    for t in range(MAX_TIMESTEPS):
        if (iter % 2000 == 0):
            img = cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR)
            cv2.imshow("Iteration " + str(iter), img)
            cv2.waitKey(50)
            
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
    
    print("Episode terminated!")
    
    # Iteration ended: Store the score, update hyperparameters
    scores[iter-1] = curr_score

    # Epsilon decay - only if we improved the reward
    epsilons[iter-1] = policy.epsilon()
    if iter > 1 and curr_score > scores[iter-2]:
        policy.decay_epsilon(iter)
        print("New probability to make a random move: " + str(policy.epsilon()))
    
    # Learning rate decay
    learn.decay_learning_rate(iter)

    # Reset environment and choose first action of next iteration
    prev_obs, info = env.reset()
    action = policy.next_action(learn._q_table, curr_obs)

# Show stats
fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Cumulative Reward')
ax1.plot(range(0, N_ITERATIONS), scores,  color)
ax1.plot(range(0, N_ITERATIONS), [0] + [np.mean(scores[:i]) for i in range(1, N_ITERATIONS)], 'tab:red')
ax1.tick_params(axis='y')

ax2 = ax1.twinx()

color = 'tab:purple'
ax2.set_ylabel('Randomness (%)')
ax2.plot(range(0, N_ITERATIONS), epsilons, color)
ax2.tick_params(axis='y')

fig.tight_layout()
plt.show()

quit()