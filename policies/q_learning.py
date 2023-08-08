import numpy as np
import random as rand

import policies.scaled_ndarray as sa

class QLearningPolicy:
    '''
        Policy which computes and maximises the Q-value(s_t, a_t) for each state and action
        env - Gymnasium environment
        n_iter - Number of iterations to perform
        alpha - Learning rate (0, 1). The higher the value, the more importance will be given to early episodes
        gamma - Discount factor (0, 1]. The higher the value, the more importance we give to rewards coming from long episodes (good start concept)
    '''
    def __init__(self, n_iter, alpha, gamma):
        self._n_iterations = n_iter
        self._alpha = alpha
        self._gamma = gamma
        self._scores = np.zeros(self._n_iterations)

    '''
        Initialize the Q-value table as a (N_1 x N_2 x ... x A)-dimensional array, where N_i is the i-th state-space dimension magnitude and A is the action_space magnitude
        state_space_shape - Tuple representing the shape of the state space
        state_space_scale - Tuple representing the discretization intervals of the state space (must have same dimension of state_space_shape)
        action_space - Tuple containing available actions
    '''
    def InitPolicy(self, state_space_shape, state_space_scale, action_space):
        self._state_space_shape = state_space_shape
        self._state_space_scale = state_space_scale
        self._action_space = action_space
        self._q_table = sa.ScaledNDArray(state_space_shape + (action_space.n,), state_space_scale + (1,))

    '''
        Update the Q_value(s, a) related to the last executed action starting from the previous state
    '''
    def update_q_value(self, prev_state, action, curr_state, reward):    
        prev_q = self._q_table[prev_state + (action,)]        # Q-value to be updated
        curr_q = self._q_table[curr_state]                    # (Q_value[L], Q_value[R])

        prev_q = (1-self._alpha)*prev_q + self._alpha*(reward + self._gamma*(max(curr_q))) 
        self._q_table[prev_state + (action,)] = prev_q
        return prev_q

    '''
        Choose next action to be executed, with epsilon-greedy policy. 
        curr_state - Current observation from which to decide the next action
        epsilon - Probability to make a random move instead of the optimal action (i.e. probability to explore instead of exploit)
    '''
    def next_action(self, curr_state, eps):
        if rand.randint(1, 100) <= eps*100:
            return self._action_space.sample()
        else:
            curr_q = self._q_table[curr_state]                    # (Q_value[L], Q_value[R])
            return np.argmax(curr_q)