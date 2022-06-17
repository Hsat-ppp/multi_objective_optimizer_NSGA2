import numpy as np


def sphere_function(data):
    assert data.ndim == 2, 'expected dim for data: {0}, but got input: {1}'.format(2, data.ndim)
    return np.sum((data+3.0)**2, axis=1)
