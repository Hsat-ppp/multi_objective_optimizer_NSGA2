import os
import random

import numpy as np

from common_settings import *
from dd_cmaes_optimizer import DD_CMAES
import functions_to_be_optimized


def create_output_file():
    # create output file
    with open('convergence_history.csv', 'w') as f:
        pass
    with open('best_solution_history.csv', 'w') as f:
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
    os.mkdir(result_dir)

    # create optimizer
    optimizer = DD_CMAES(functions_to_be_optimized.sphere_function)

    # optimization iteration
    for g in range(generation_max):
        with open('Generation.csv', 'w') as f:
            print(g + 1, file=f)
        optimizer.proceed_generation()
        print('generation:', optimizer.generation, sep=' ')
        print('best solution:', optimizer.best_solution, sep='\n')
        print('best evaluation:', optimizer.best_eval, sep=' ')
        with open('convergence_history.csv', 'a') as f:
            print(g + 1, optimizer.num_of_evaluation, optimizer.best_eval, sep=',', file=f)
        with open('best_solution_history.csv', 'a') as f:
            print(*optimizer.best_solution, sep=',', file=f)

    print('optimization end.')
    print('best solution:', optimizer.best_solution, sep='\n')
    print('best evaluation:', optimizer.best_eval, sep=' ')


if __name__ == '__main__':
    optimize()
