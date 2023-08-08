import random as rand
import numpy as np

class EpsilonGreedyPolicy:
    '''
        Epsilon-greedy policy. Get the optimal action based on some action-value function, with a chance to choose a random action instead of the optimal one.
        epsilon - Probability to make a random move instead of the optimal action (i.e. probability to explore instead of exploit)
    '''
    def __init__(self, action_space, eps):
        self._action_space = action_space
        self._eps = eps

    '''
        Get the current epsilon value
    '''
    def epsilon(self):
        return self._eps

    '''
        Set a new epsilon
        new_epsilon - New probability to make a random move instead of the optimal action (i.e. probability to explore instead of exploit)
    '''
    def set_epsilon(self, new_eps):
        self._eps = new_eps

    '''
        Sample a random action between all those of the action space
    '''
    def random_action(self):
        return self._action_space.sample()

    '''
        Choose next action to be executed 
        curr_state - Current observation from which to decide the next action
    '''
    def next_action(self, q_table, curr_state):
        if rand.randint(1, 100) <= self._eps*100:
            return self.random_action()
        else:
            curr_q = q_table[curr_state]                    # (Q_value[a_1], Q_value[a_2], ...)
            return np.argmax(curr_q)