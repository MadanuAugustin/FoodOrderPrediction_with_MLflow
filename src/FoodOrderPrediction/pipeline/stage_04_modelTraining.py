

from src.FoodOrderPrediction.config.configuration import ConfigurationManager
from src.FoodOrderPrediction.components.model_training_com4 import ModelTrainer



STAGE_NAME = 'Model Trainer Stage'

class ModelTrainingPipeline:
    def __init__(self):
        pass


    def main(self):

        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer_config = ModelTrainer(config = model_trainer_config)
        model_trainer_config.initiate_model_training()