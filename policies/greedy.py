import random as rand
import numpy as np

class GreedyPolicy:
    '''
        Epsilon-greedy policy. Get the optimal action based on some action-value function, with a chance to choose a random action instead of the optimal one.
        epsilon - Probability to make a random move instead of the optimal action (i.e. probability to explore instead of exploit)
    '''
    def __init__(self, action_space):
        self._action_space = action_space

    '''
        Choose next action to be executed as the one with highest Q-value
        curr_state - Current observation from which to decide the next action
    '''
    def next_action(self, q_table, curr_state):
        curr_q = q_table[curr_state]                    # (Q_value[a_1], Q_value[a_2], ...)
        return np.argmax(curr_q)