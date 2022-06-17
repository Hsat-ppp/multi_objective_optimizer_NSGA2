import logging
import random

import numpy as np

from single_objective_optimizer.model.common_settings import *
from single_objective_optimizer.model.dd_cmaes_optimizer import DDCMAES
from single_objective_optimizer.model import functions_to_be_optimized

logger = logging.getLogger('info_logger')


def create_output_file():
    # create output file
    with open('convergence_history.csv', 'w'):
        pass
    with open('best_solution_history.csv', 'w'):
        pass


def optimize(seed_num=None):
    # fix seed number and save
    if seed_num is None:
        seed_num = np.random.randint(0, (2**30)-1)
    random.seed(seed_num)
    np.random.seed(seed_num)
    with open('seed_num.csv', 'w') as f:
        print(seed_num, sep=',', file=f)

    # create output file and folder
    create_output_file()

    # create optimizer
    optimizer = DDCMAES(functions_to_be_optimized.sphere_function)

    # optimization iteration
    for g in range(generation_max):
        with open('Generation.csv', 'w') as f:
            print(g + 1, file=f)
        optimizer.proceed_generation()
        logger.info('generation {} end'.format(g + 1))
        with open('convergence_history.csv', 'a') as f:
            print(g + 1, optimizer.num_of_evaluation, optimizer.best_eval, sep=',', file=f)
        with open('best_solution_history.csv', 'a') as f:
            print(*optimizer.best_solution, sep=',', file=f)

    logger.info('optimization end.')
