import logging
import os
import random

import numpy as np


def set_seed_num(seed_num):
    if seed_num is None:
        seed_num = np.random.randint(0, (2 ** 30) - 1)
    np.random.seed(seed_num)
    random.seed(seed_num)
    os.environ['PYTHONHASHSEED'] = str(seed_num)
    with open('seed_num.csv', 'w') as f:
        print(seed_num, sep=',', file=f)
