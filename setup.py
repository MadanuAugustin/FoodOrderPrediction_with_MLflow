

## setup file looks for the folders which are having __init__ constructor and converts them as local packages
## after local packages are created we can import the functions from the files


from setuptools import setup, find_packages


with open("README.md", 'r', encoding='utf-8') as f:
    long_description = f.read()


setup(
    name="FoodOrderPrediction",
    version='0.0.0.0',
    author = 'augustin',
    author_email='augustinmadanu7@gmail.com',
    description='End-end-machine-learning-project-with-MLflow',
    long_description=long_description,
    url='https://github.com/MadanuAugustin/FoodOrderPrediction_with_MLflow.git',
    packages= find_packages()
)