import argparse
import json
import logging

import tqdm

from multi_objective_optimizer.model.settings import *
from multi_objective_optimizer.model.EAPES import EAPES
from multi_objective_optimizer.model import functions_to_be_optimized
from multi_objective_optimizer.utils.utils import set_seed_num

logger = logging.getLogger('info_logger')


def get_argparser_options():
    parser = argparse.ArgumentParser(description='''
                                    This is a multi objective optimizer based on NSGA2.
                                    ''')
    parser.add_argument('-g', '--num_of_generations', default=200, type=int,
                        help='number of generations (iterations)')
    parser.add_argument('-fp', '--feasible_population_size', default=n*20, type=int,
                        help='feasible population size or number of individuals.')
    parser.add_argument('-ip', '--infeasible_population_size', default=n*20, type=int,
                        help='infeasible population size or number of individuals.')
    parser.add_argument('-c', '--children_size', default=n*5, type=int,
                        help='population size or number of individuals.')
    parser.add_argument('-s', '--seed_num', type=int,
                        help='seed number for reproduction.')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='if set, we disables progress bar.')
    args = parser.parse_args()
    return args


def optimize():
    # get args
    args = get_argparser_options()
    assert args.num_of_generations >= 1, 'Option "num_of_generations" need to be positive. ' \
                                         'Got: {}'.format(args.num_of_generations)
    assert args.feasible_population_size >= 1, 'Option "feasible_population_size" need to be positive. ' \
                                      'Got: {}'.format(args.feasible_population_size)
    assert args.infeasible_population_size >= 1, 'Option "infeasible_population_size" need to be positive. ' \
                                      'Got: {}'.format(args.infeasible_population_size)
    assert args.children_size >= 1, 'Option "children_size" need to be positive. ' \
                                      'Got: {}'.format(args.children_size)
    # save args
    logger.info('args options')
    logger.info(args.__dict__)
    with open('args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    # fix seed number and save
    set_seed_num(args.seed_num)

    # create optimizer
    optimizer = EAPES(Fnpop=args.feasible_population_size, Inpop=args.infeasible_population_size,
                      nc=args.children_size,
                      eval_func=functions_to_be_optimized.SRN,
                      constraint_func=functions_to_be_optimized.SRN_constraint,
                      init_range_lower=-20, init_range_upper=20,
                      boundary_lower=-20, boundary_upper=20
                      )
    optimizer.neutralization()

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
        optimizer.step()
        with open(result_dir + 'Generation' + str(g + 1) + '/Pareto.csv', 'w') as f:
            for p in optimizer.Pareto:
                for m in reversed(range(nobj)):
                    if m == 0:
                        s = '\n'
                    else:
                        s = ','
                    print(str(p[-1*(m+5+1)]), sep='', end=s, file=f)
        with open(result_dir + 'Generation' + str(g + 1) + '/Pareto_gene.csv', 'w') as f:
            for p in optimizer.Pareto:
                s = ""
                for i in range(n):
                    s += str(p[i])
                    if not i == n-1:
                        s += ","
                print(s, file=f)
        with open(result_dir + 'Generation' + str(g + 1) + '/Pareto_constraint.csv', 'w') as f:
            for p in optimizer.Pareto:
                print(p[-1], file=f)

    logger.info('Optimization end.')
