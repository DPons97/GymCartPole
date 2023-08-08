import gymnasium as gym
from gymnasium.wrappers import TransformObservation
import math
import matplotlib.pyplot as plt

import math_utils as mu
import policies.q_learning as qlearning

# Globals
N_ITERATIONS = 2000     # Number of iterations (simulations)
ALPHA = 0.1             # Learning rate
GAMMA = 0.6             # Discount factor
EPS = 0.995             # Epsilon-greedy starting randomness

# Initialize the environment
env = gym.make("CartPole-v1")#, render_mode = "human")

# Transforming observation so that:
# Position: 2.3344 -> 2.3, steps of 0.1 [-0.1, 0, 0.1, 0.2, 0.3, ...]
# Cart Velocity ignored for now
# Pole Angle: 0.1254 -> 0.1221 steps of (1 deg_to_rad) [-0.017, 0, 0.017, 0.034, ...]
# Pole angular Velocity ignored for now
state_space_shape = (94, 50, 48, 100)
state_space_scale = (0.1, 0.1, math.radians(1), 0.1)
env = TransformObservation(env, lambda obs: (mu.round_to(obs[0], state_space_scale[0]), mu.round_to(obs[1], state_space_scale[1], 1), mu.round_to(obs[2], state_space_scale[2], 4), mu.round_to(obs[3], state_space_scale[3], 1)))

# Initialize policy
policy = qlearning.QLearningPolicy(n_iter=N_ITERATIONS, alpha=ALPHA, gamma=GAMMA)
policy.InitPolicy(state_space_shape=state_space_shape, state_space_scale=state_space_scale, action_space=env.action_space)

prev_obs, info = env.reset()
action = env.action_space.sample()
curr_score = 0
eps = EPS
for i in range(N_ITERATIONS):
    curr_obs, reward, term, trunc, info = env.step(action)
    curr_score += reward
    print("New observation: " + str(curr_obs))

    #if (term):
        # Update policy with negative reward
    #    reward = -1

    new_q_value = policy.update_q_value(prev_state=prev_obs, action=action, curr_state=curr_obs, reward=reward)
    print("New q value for " + str(prev_obs + (action,)) + " = " + str(new_q_value))
    
    if (term):
        policy._scores[i] = curr_score
        curr_score = 0
        prev_obs, info = env.reset()
        eps *= EPS 
        print("Episode terminated! Starting " + str(i+1) + "th iteration...")
        print("New probability to make a random move: " + str(EPS))
    else:
        prev_obs = curr_obs
    
    action = policy.next_action(curr_obs, eps)
    print("Next action will be: " + ("L" if action==0 else "R"))

# Show stats
fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Cumulative Reward')
ax1.plot(range(0, N_ITERATIONS), policy._scores, color)
ax1.tick_params(axis='y')

ax2 = ax1.twinx()

color = 'tab:red'
ax2.set_ylabel('Randomness (%)')
ax2.plot(range(0, N_ITERATIONS), [pow(EPS, i)  for i in range(0, N_ITERATIONS)], color=color)
ax2.tick_params(axis='y')

fig.tight_layout()
plt.show()

quit()