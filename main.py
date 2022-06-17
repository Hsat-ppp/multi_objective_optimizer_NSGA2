import json
import logging
import logging.config

import single_objective_optimizer.controller.controller

# load logging config
with open('log_config.json', 'r') as f:
    log_conf = json.load(f)
logging.config.dictConfig(log_conf)

if __name__ == '__main__':
    # run optimization
    single_objective_optimizer.controller.controller.optimize()
