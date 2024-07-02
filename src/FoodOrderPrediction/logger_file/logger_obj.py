

import os
import sys
import logging

logging_str = "[%(asctime)s : %(levelname)s : %(module)s : %(message)s]"

log_dir = 'logs'

log_filepath = os.path.join(log_dir, "running_logs.log")

os.makedirs(log_dir, exist_ok=True)



logging.basicConfig(
    level=logging.INFO,
    format=logging_str,


    handlers=[
        ## filehandler creates the log folder and saves all the logs
        ## streamhandler will print the log in the terminal
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)

    ]
)

logger = logging.getLogger('FoodOrderPrediction_with_MLflow')

