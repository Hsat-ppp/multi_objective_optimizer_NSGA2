import numpy as np


def crossover_sbx(parent1, parent2, eta_d):
    parent1 = np.array(parent1)
    parent2 = np.array(parent2)
    child1 = np.zeros_like(parent1)
    child2 = np.zeros_like(parent2)
    for j in range(parent1.shape[0]):
        u = np.random.rand()
        if u <= 0.5:
            beta = (2 * u) ** (1.0 / (eta_d + 1))
        else:
            beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta_d + 1))
        child1[j] = (0.5 * ((1 + beta) * parent1[j] + (1 - beta) * parent2[j]))
        child2[j] = (0.5 * ((1 - beta) * parent1[j] + (1 + beta) * parent2[j]))
    return child1, child2


def mutation_gaussian(ind, probability):
    ind = np.array(ind)
    for i in range(ind.shape[0]):
        if np.random.rand() <= probability:
            ind[i] += np.random.randn()
    return ind
