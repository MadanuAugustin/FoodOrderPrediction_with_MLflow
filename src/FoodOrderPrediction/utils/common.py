


## utils file is used for storing the common functionalities in the project without re-writing the code thus saving memory and time.
## re-usability of code

import os
import sys
# from box.exceptions import BoxValueError
import yaml
from src.FoodOrderPrediction.logger_file.logger_obj import logger
from src.FoodOrderPrediction.Exception.custom_exception import CustomException
import json
import joblib
from box import ConfigBox
from pathlib import Path
from typing import Any


# read_yaml function is used for reading yaml files

def read_yaml(path_to_yaml : Path) -> ConfigBox:
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file : {path_to_yaml} loaded successfully...!")
            return ConfigBox(content)
    except Exception as e:
        raise CustomException(e, sys)
    

# create_directories function is used to create directories

def create_directories(path_to_directories : list, verbose = True):
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)

        if verbose:
            logger.info(f'created directory at : {path}')
    

# save_json function is used to save files in the json format

def save_json(path : Path, data : dict):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    
    logger.info(f'json file saved at : {path}')



# load_json function is used to load the existing json files

def load_json(path : Path) -> ConfigBox:
    with open(path) as f:
        content = json.load(f)

    logger.info(f'json file loaded successfully from : {path}')

    return ConfigBox(content)


# save_bin function is used to save the files in binary format

def save_bin(data : Any, path : Path):
    joblib.dump(value= data, filename=path)
    logger.info(f"binary file saved at : {path}")


# load_bin function is used to load existing binary files

def load_bin(path : Path) -> Any:
    data = joblib.load(path)
    logger.info(f'binary file loaded from {path}')
    return data


# get_size function is used to get the size of the files

def get_size(path : Path) -> str:
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~{size_in_kb} KB"
