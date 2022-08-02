import json
import logging
import logging.config

# import multi_objective_optimizer.controller.controller
import multi_objective_optimizer.controller.controller_EAPES

# load logging config
with open('log_config.json', 'r') as f:
    log_conf = json.load(f)
logging.config.dictConfig(log_conf)

if __name__ == '__main__':
    # run optimization
    # multi_objective_optimizer.controller.controller.optimize()
    multi_objective_optimizer.controller.controller_EAPES.optimize()
