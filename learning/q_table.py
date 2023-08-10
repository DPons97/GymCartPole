import numpy as np
import json
from json import JSONEncoder

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

class QTable:
    '''
        Initialize Q_table with random values in range (0,1)
        shape - Tuple representing the shape of the state space (e.g. (4, 4, 2) is a 4x4x2 dimensions array)
        scale_value - Tuple representing the intervals between one value of each dimension and the next one (e.g. [0 0.1 0.2 0.3 ...] has scale_value 0.1)
    '''
    def __init__(self, shape=(0,), scale_value=(0,)):
        self._shape = shape
        self._scale_value = scale_value
        self._table = np.random.uniform(low=0, high=1, size=shape)

    '''
        Get a tuple to index a specific element of this array, and return a tuple with the actual integer indices
    '''
    def key_to_idx(self, key):
        return tuple(round(k / s) for k, s in zip(key, self._scale_value))

    ''' 
        Operator []
        key - Tuple representing the key to access the state space
        e.g. (0.1, 5, 2.4) can be used to access a 3-dimensional state-space
    '''
    def __getitem__(self, key):
        return self._table[self.key_to_idx(key)]
    
    def __setitem__(self, key, value):
        self._table[self.key_to_idx(key)] = value

    '''
        Save the current Q_table to a json file
    '''
    def to_json_file(self, path='q_table.json'):
        data = {"shape" : self._shape, "scale" : self._scale_value,  "q_table" : self._table}
        with open(path, "w") as file:
            json.dump(data, file, cls=NumpyArrayEncoder)
        print("Q-table saved at " + path)

    @staticmethod
    def from_json_file(path):
        try:
            with open(path, "r") as read_file:
                decodedData = json.load(read_file)
                newTable = QTable(decodedData["shape"], decodedData["scale"])
                newTable._table = np.asarray(decodedData["q_table"])
            return newTable
        except:
            return None
        
        
    
