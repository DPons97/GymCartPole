import random as rand
import numpy as np

class EpsilonGreedyPolicy:
    '''
        Epsilon-greedy policy. Get the optimal action based on some action-value function, with a chance to choose a random action instead of the optimal one.
        epsilon - Probability to make a random move instead of the optimal action (i.e. probability to explore instead of exploit)
    '''
    def __init__(self, action_space, starting_eps, decay_eps, start_decay_iter=0):
        self._action_space = action_space
        self._eps = starting_eps
        self._decay_eps = decay_eps
        self._start_decay_iter = start_decay_iter

    '''
        Get the current epsilon value
    '''
    def epsilon(self):
        return self._eps

    '''
        Get decayed epsilon value
        iter - iteration at which to get the decayed epsilon value
    '''
    def decay_epsilon(self, iter):
        if (self._eps > 0.05 and iter > self._start_decay_iter):
            self._eps = pow(self._decay_eps, iter - self._start_decay_iter)

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