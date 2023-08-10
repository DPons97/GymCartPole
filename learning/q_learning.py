import learning.q_table as qt

class QLearning:
    '''
        Off-Policy learning algorithm which computes and maximises the Q-value(s_t, a_t) for each state and action
        env - Gymnasium environment
        alpha - Learning rate (0, 1). The higher the value, the more importance will be given to early episodes
        gamma - Discount factor (0, 1]. The higher the value, the more importance we give to rewards coming from long episodes (good start concept)
    '''
    def __init__(self, alpha, gamma):
        self._start_alpha = alpha
        self._alpha = alpha
        self._gamma = gamma

    '''
        Initialize the Q-value table as a (N_1 x N_2 x ... x A)-dimensional array, where N_i is the i-th state-space dimension magnitude and A is the action_space magnitude
        state_space_shape - Tuple representing the shape of the state space
        state_space_scale - Tuple representing the discretization intervals of the state space (must have same dimension of state_space_shape)
        action_space - Tuple containing available actions
    '''
    def InitQTable(self, state_space_shape, state_space_scale, action_space):
        self._state_space_shape = state_space_shape
        self._state_space_scale = state_space_scale
        self._q_table = qt.QTable(state_space_shape + (action_space.n,), state_space_scale + (1,))

    '''
        Update the Q_value(s, a) related to the last executed action starting from the previous state
    '''
    def update_q_value(self, prev_state, action, curr_state, reward):    
        prev_q = self._q_table[prev_state + (action,)]        # Q-value to be updated
        curr_q = self._q_table[curr_state]                    # (Q_value[L], Q_value[R])

        prev_q = (1-self._alpha)*prev_q + self._alpha*(reward + self._gamma*(max(curr_q))) 
        self._q_table[prev_state + (action,)] = prev_q
        return prev_q
    
    def decay_learning_rate(self, iter):
        self._alpha = pow(self._start_alpha, iter)