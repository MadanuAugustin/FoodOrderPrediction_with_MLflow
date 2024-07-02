

### template file is responsible for creating the directories and files required for the project.


import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] : %(message)s : ')

### defining the project name

project_name = 'FoodOrderPrediction'



list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/entity/config_entity.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "params.yaml",
    "schema.yaml",
    "main.py",
    "app.py",
    "Dockerfile",
    "requirements.txt",
    "setup.py",
    "research/trails.ipynb",
    'templates/index.html',
    "templates/results.html"
]


for filepath in list_of_files:

    # windows considers the backward slash instead of forward slash for creating the folders to overcome such issues we are using--
    # Path function to create windows paths

    filepath = Path(filepath)

    #our list_of_files consists of folders and files as a path which needed to be split separately as folders and files

    filedir, filename = os.path.split(filepath)

    if filedir !="":
        
        ## creating directories

        os.makedirs(filedir, exist_ok=True)

        logging.info(f"creating directory : {filedir} for the file : {filename}")


    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):

        ## creating files

        with open(filepath, 'w') as f:
            pass
            logging.info(f'creating empty file : {filepath}')


    else:
        logging.info(f'{filename} is already exists')

