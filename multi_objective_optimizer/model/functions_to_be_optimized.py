import os

from multi_objective_optimizer.model.common_settings import *
import numpy as np


def DTLZ2(data):
    """DRLZ2 function
    x : [0, 1]
    """
    with open('Generation.csv', 'r') as f:
        g = int(f.readline())
    os.mkdir(result_dir + 'Generation' + str(g))

    evals = np.ones((data.shape[0], nobj))
    g = np.sum((data[:, nobj-1:] - 0.5)**2, axis=1)
    for m in range(nobj):
        evals[:, m] *= (1 + g)
        for i in range(nobj - m - 1):
            evals[:, m] *= np.cos(0.5 * data[:, i] * np.pi)
        if m != 0:
            evals[:, m] *= np.sin(0.5 * data[:, nobj - m - 1] * np.pi)
    return evals


def zero_constraint(data):
    """for functions with no constraints
    """
    return np.zeros(data.shape[0])


def SRN(data):
    """SRN function
    data: [-20, 20]
    only assuming n=2 and nobj=2
    """
    assert data.shape[1] == 2, 'SRN function only accepts 2-D input.'
    assert nobj == 2, 'SRN function only accepts 2 objectives.'
    with open('Generation.csv', 'r') as f:
        g = int(f.readline())
    os.mkdir(result_dir + 'Generation' + str(g))

    evals = np.zeros((data.shape[0], nobj))
    evals[:, 0] = (data[:, 0] - 2)**2 + (data[:, 1] - 1)**2 + 2
    evals[:, 1] = 9 * data[:, 0] - (data[:, 1] - 1)**2
    return evals


def SRN_constraint(data):
    """SRN function
    data: [-20, 20]
    only assuming n=2 and nobj=2
    """
    evals = np.zeros(data.shape[0])
    # 違反量を測るので，マイナスは0に丸める
    evals += np.max(np.stack([np.zeros(data.shape[0]), ((data[:, 0]**2 + data[:, 1]**2) - 255)], axis=1), axis=1)
    evals += np.max(np.stack([np.zeros(data.shape[0]), ((data[:, 0] - 3 * data[:, 1]) - (-10))], axis=1), axis=1)
    return evals
