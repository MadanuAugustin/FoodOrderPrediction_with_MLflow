

from src.FoodOrderPrediction.config.configuration import ConfigurationManager
from src.FoodOrderPrediction.components.model_evaluation_com5 import ModelEvaluation



STAGE_NAME = 'Model Evaluation Stage'


class ModelEvaluationPipeline:
    def __init__(self):
        pass


    def main(self):

        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation_config = ModelEvaluation(config = model_evaluation_config)
        model_evaluation_config.log_into_mlflow()