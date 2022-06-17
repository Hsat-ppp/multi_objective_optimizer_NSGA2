import numpy as np


def sphere_function(data):
    return np.sum((data+3.0)**2, axis=1)
