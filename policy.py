import numpy as np

class Policy:
    '''
        shape - Tuple representing the shape of the state space (e.g. (4, 4, 2) is a 4x4x2 dimensions array)
        scale_value - Tuple representing the intervals between one value of each dimension and the next one (e.g. [0 0.1 0.2 0.3 ...] has scale_value 0.1)
    '''
    def __init__(self, shape, scale_value):
        self._state_space = np.ndarray(shape)
        self._scale_value = scale_value

    ''' 
        Operator []
        key - Tuple representing the key to access the state space
        e.g. (0.1, 5, 2.4) can be used to access a 3-dimensional state-space
    '''
    def __getitem__(self, key):
        idx = tuple(round(k / s) for k, s in zip(key, self._scale_value))
        return self._state_space[idx]