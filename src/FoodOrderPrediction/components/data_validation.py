import sys
import pandas as pd
from src.FoodOrderPrediction.entity.config_entity import DataValidationConfig
from src.FoodOrderPrediction.logger_file.logger_obj import logger
from src.FoodOrderPrediction.Exception.custom_exception import CustomException


class DataValidation:
    def __init__(self, config : DataValidationConfig):
        self.config = config



    def validate_all_columns(self) -> bool:
        try:
            validation_status = None

            data = pd.read_csv(self.config.local_data_file)

            all_cols = list(data.columns)

            all_schema = self.config.all_schema.keys()

            for col in all_cols:
                if col not in all_schema:
                    validation_status = False
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"validation status : {validation_status}")
                
                else:
                    validation_status = True
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f'validation status : {validation_status}')

            return validation_status
        
        except Exception as e:
            raise CustomException(e, sys)