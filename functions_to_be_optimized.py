import numpy as np

from common_settings import *


def sphere_function(data):
    return np.sum((data+3.0)**2, axis=1)
