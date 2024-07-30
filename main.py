import gymnasium as gym
from gymnasium.wrappers import TransformObservation
import math
import numpy as np

import math_utils as mu
import learning.q_learning as qlearning
import policies.epsilon_greedy as ep
import policies.greedy as gp
import rl_utils as rlu

def run(render = False):
    # Globals
    N_ITERATIONS = 10000        # Number of iterations (simulations)
    ALPHA = 0.1                   # Learning rate, constant for first 1000 iterations
    GAMMA = 0.99                # Discount factor
    DECAY_EPS_FACTOR = 0.0001  # Epsilon-greedy starting randomness, constant for first 1000 iterations

    # Initialize the environment
    env = gym.make("CartPole-v1", render_mode= "rgb_array" if render else None)

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

    # Initialize learning algorithm
    learn = qlearning.QLearning(gamma=GAMMA, starting_alpha=ALPHA)
    learn.init_q_table(state_space_shape=state_space_shape, action_space=env.action_space)

    # Initialize policy for learning
    learning_policy = ep.EpsilonGreedyPolicy(action_space = env.action_space, decay_factor = DECAY_EPS_FACTOR)

    # Initialize policy for testing
    testing_policy = gp.GreedyPolicy(action_space = env.action_space)

    # Init stats data
    scores = []
    mean_scores = []
    epsilons = []

    # First action of first iteration is always random
    iter = 0
    while True:
        # Run learning algorithm
        rlu.Learn(env = env, algo = learn, policy = learning_policy, iteration = iter)
        
        # Evaluate performance of new version of learning algorithm with testing environment
        score = rlu.Test(env = env, algo = learn, policy = testing_policy, iter = iter)

        # Iteration ended: Store the score, update hyperparameters
        scores.append(score)

        # Epsilon decay - only if we improved the reward
        epsilons.append(learning_policy.epsilon())
        if iter > 1 and score > scores[iter-2]:
            learning_policy.decay_epsilon()
        
        # Evaluate termination condition
        mean_scores = np.mean(scores[len(scores)-100:])

        if (iter % 100 == 0):
            print(f'Iteration: {iter} {score}     Epsilon: {epsilons[iter]:0.2f}   Mean Rewards: {mean_scores:0.1f}')
        
        if (mean_scores > 1000):
            print(f'Iteration: {iter} {score}     Epsilon: {epsilons[iter]:0.2f}   Mean Rewards: {mean_scores:0.1f}')
            break

        iter += 1

    rlu.plot_stats(len(scores), scores, epsilons)
    print(f'Finished in {iter} iteration.   Epsilon: {epsilons[iter]:0.2f}   Mean Rewards: {mean_scores:0.1f}')

    import os
    learn.save(os.getcwd() + "/tables/cart_q_table.json")

    quit()

if __name__ == "__main__":
    run(True)