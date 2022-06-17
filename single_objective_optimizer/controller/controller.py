import argparse
import logging
import random

import numpy as np
import tqdm

from single_objective_optimizer.model.common_settings import *
from single_objective_optimizer.model.dd_cmaes_optimizer import DDCMAES
from single_objective_optimizer.model import functions_to_be_optimized

logger = logging.getLogger('info_logger')


def get_argparser_options():
    parser = argparse.ArgumentParser(description='''
                                    This is a single objective optimizer 
                                    based on CMA-ES and dd-CMA-ES, its advanced version.
                                    ''')
    parser.add_argument('-g', '--num_of_generations', default=100, type=int,
                        help='number of generations (iterations)')
    parser.add_argument('-p', '--population_size', default=int(4+np.floor(3*np.log(n))), type=int,
                        help='population size or number of individuals.')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='if set, we disables progress bar.')
    args = parser.parse_args()
    return args


def create_output_file():
    # create output file
    with open('convergence_history.csv', 'w'):
        pass
    with open('best_solution_history.csv', 'w'):
        pass


def optimize(seed_num=None):
    # get args
    args = get_argparser_options()
    assert args.num_of_generations >= 1, 'Option "num_of_generations" need to be positive. ' \
                                         'Got: {}'.format(args.num_of_generations)
    assert args.population_size >= 1, 'Option "population_size" need to be positive. ' \
                                      'Got: {}'.format(args.population_size)

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
    optimizer = DDCMAES(functions_to_be_optimized.sphere_function,
                        population_size=args.population_size)

    # optimization iteration
    logger.info('Optimization start.')
    iterator = None
    if args.quiet:
        iterator = range(args.num_of_generations)
    else:
        iterator = tqdm.tqdm(range(args.num_of_generations))
    for g in iterator:
        with open('Generation.csv', 'w') as f:
            print(g + 1, file=f)
        optimizer.proceed_generation()
        with open('convergence_history.csv', 'a') as f:
            print(g + 1, optimizer.num_of_evaluation, optimizer.best_eval, sep=',', file=f)
        with open('best_solution_history.csv', 'a') as f:
            print(*optimizer.best_solution, sep=',', file=f)

    logger.info('''Optimization end.
                Total generations: {0}
                Best eval: {1}
                '''.format(args.num_of_generations, optimizer.best_eval))
