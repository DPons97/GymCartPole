import numpy as np

class QTable:
    '''
        Initialize Q_table with random values in range (0,1)
        shape - Tuple representing the shape of the state space (e.g. (4, 4, 2) is a 4x4x2 dimensions array)
        scale_value - Tuple representing the intervals between one value of each dimension and the next one (e.g. [0 0.1 0.2 0.3 ...] has scale_value 0.1)
    '''
    def __init__(self, shape, scale_value):
        self.shape = shape
        self._scale_value = scale_value
        self._state_space = np.random.uniform(low=0, high=1, size=shape)

    '''
        Get a tuple to index a specific element of this array, and return a tuple with the actual integer indices
    '''
    def KeyToIdx(self, key):
        return tuple(round(k / s) for k, s in zip(key, self._scale_value))

    ''' 
        Operator []
        key - Tuple representing the key to access the state space
        e.g. (0.1, 5, 2.4) can be used to access a 3-dimensional state-space
    '''
    def __getitem__(self, key):
        return self._state_space[self.KeyToIdx(key)]
    
    def __setitem__(self, key, value):
        self._state_space[self.KeyToIdx(key)] = value