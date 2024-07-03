
import os
import sys
import numpy as np
import pandas as pd
import joblib
from src.FoodOrderPrediction.logger_file.logger_obj import logger
from src.FoodOrderPrediction.Exception.custom_exception import CustomException
from src.FoodOrderPrediction.entity.config_entity import DataTransformationConfig
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder





class DataTransformation:
    def __init__(self, config : DataTransformationConfig):
        
        self.config = config




    def data_split(self):
        try:

            logger.info(f'-----------Entered data_split method---------------')
        
            data = pd.read_csv(self.config.local_data_file)

            data = data[["Age", "Gender", "Marital_status", "occupation", 
                   "monthly_income", "educational_qualifications", 
                   "family_size", "pincode", "feedback", "unknow_column"]]

            data.rename(columns={'unknow_column' : 'Target'}, inplace=True)

            data["monthly_income"] = data["monthly_income"].map({"No Income": 0, 
                                                     "25001 to 50000": 50000, 
                                                     "More than 50000": 70000, 
                                                     "10001 to 25000": 25000,
                                                     "Below Rs.10000": 10000})

            
            data["educational_qualifications"] = data["educational_qualifications"].map({"Graduate": 3, 
                                                                             "Post Graduate": 4, 
                                                                             "Ph.D": 5, "School": 2,
                                                                             "Uneducated": 1})
            

            train, test = train_test_split(data, test_size=0.2, random_state=42)

            train.to_csv(os.path.join(self.config.train_path, 'train.csv'), index = False, header = True)
        
            test.to_csv(os.path.join(self.config.test_path, 'test.csv'), index = False, header = True)

            logger.info(f'----------------saved train test data in csv format------------------')

            logger.info(f'------------The shape of the train data is {train.shape}')

            logger.info(f'--------------The shape of the test data is {test.shape}')

            logger.info(f'------------completed data splitting----------------------')

        except Exception as e:
            raise CustomException(e, sys)
        


    def preprocessor_fun(self):
        try:

            logger.info(f'---------------Entered preprocessor function------------------')


            numeric_columns = ['Age', 'monthly_income', 'family_size', 'pincode']
            
            categoric_columns = ['Gender', 'Marital_status', 'occupation', 'feedback']

            logger.info(f'----------creating transformer pipelines---------------')

            numeric_pipeline = Pipeline(
                steps=[
                    ('standardscaler', StandardScaler(with_mean=True))
                ]
            )

            categoric_pipeline = Pipeline(
                steps=[
                    ('onehotencoder', OneHotEncoder(drop='first'))
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ('numericpipeline', numeric_pipeline, numeric_columns),
                    ('categoricpipeline', categoric_pipeline, categoric_columns)
                ]
            )

            logger.info(f'---------------completed creating transformer pipelines---------------')

            logger.info(f'--------------completed preprocessor function------------------')

            return preprocessor
            

        except Exception as e:
            raise CustomException(e, sys)
        


    def initiate_data_transformation(self):

        try:

            logger.info(f'------------started initiate_data_transformation method------------')

            train_df = pd.read_csv('artifacts//data_transformation//train.csv')
            
            test_df = pd.read_csv('artifacts//data_transformation//test.csv')

            logger.info(f'----------obtaining the preprocessor obj-----------')

            independent_train_X = train_df.drop(columns = ['educational_qualifications','Target'], axis = 1)
            dependent_train_Y = train_df[['educational_qualifications','Target']]

            independent_test_X = test_df.drop(columns = ['educational_qualifications','Target'], axis = 1)
            dependent_test_Y = test_df[['educational_qualifications','Target']]

            preprocessor_obj = self.preprocessor_fun()

            transformed_train_df = preprocessor_obj.fit_transform(independent_train_X)

            joblib.dump(preprocessor_obj, os.path.join(self.config.root_dir, 'preprocessor_obj.joblib'))

            transformed_test_df = preprocessor_obj.transform(independent_test_X)

            transformed_train_df = pd.DataFrame(np.c_[transformed_train_df, dependent_train_Y])

            transformed_test_df = pd.DataFrame(np.c_[transformed_test_df, dependent_test_Y])

            transformed_train_df.rename(columns={12 : 'Target'}, inplace=True)

            transformed_test_df.rename(columns={12 : 'Target'}, inplace=True)

            transformed_train_df.to_csv(os.path.join(self.config.root_dir, 'transformed_train_df.csv'), index = False, header = True)

            transformed_test_df.to_csv(os.path.join(self.config.root_dir, 'transformed_test_df.csv'), index = False, header = True)

            logger.info(f'-------------transformed data using preprocessor obj and saved in csv format----------')

            logger.info(f'--------------completed initiate_data_transformation method--------------')

        except Exception as e:
            raise CustomException(e, sys)