




import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from src.FoodOrderPrediction.logger_file.logger_obj import logger
from src.FoodOrderPrediction.Exception.custom_exception import CustomException


class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load(Path('model//model.joblib'))
        self.preprocessorObj = joblib.load(Path('model//preprocessor_obj.joblib'))


    # the below method takes the data from the user to predict

    def predictDatapoint(self, data):
        
        try:

            data_df = data.rename(columns = {0 : 'Age', 1 : 'monthly_income', 2 : 'family_size',
                                             3 : 'pincode', 4 : 'Gender', 5 : 'Marital_status',
                                             6 : 'occupation', 7 : 'feedback',
                                             8 : 'educational_qualifications'
                                             })
            
            data_df["monthly_income"] = data_df["monthly_income"].map({"No Income": 0, 
                                                     "25001 to 50000": 50000, 
                                                     "More than 50000": 70000, 
                                                     "10001 to 25000": 25000,
                                                     "Below Rs.10000": 10000})
            
            data_df["educational_qualifications"] = data_df["educational_qualifications"].map({"Graduate": 3, 
                                                                             "Post Graduate": 4, 
                                                                             "Ph.D": 5, "School": 2,
                                                                             "Uneducated": 1})
            
            print(data_df)

            user_numeric_cols = data_df.drop(columns = ['educational_qualifications'], axis = 1)

            user_categoric_cols = data_df[['educational_qualifications']]

            transformed_numeric_cols = self.preprocessorObj.transform(user_numeric_cols)

            # transformed_categoric_cols = user_categoric_cols['Credit_Mix'].map({'Good': 1, 'Standard' : 2, 'Bad' : 0})

            transformed_user_input = pd.DataFrame(np.c_[transformed_numeric_cols, user_categoric_cols])

            logger.info(f'---------Below is the transformed user input----------------')

            print(transformed_user_input)


            prediction = self.model.predict(transformed_user_input)

            print(prediction)

            list_output  = []

            if prediction == ['Yes\r']:
                list_output.append('Yes')
            elif prediction == ['No\r']:
                list_output.append('No')

            logger.info(f'-----------Below output is predicted by the model---------------')

            print(list_output)

            return list_output
        
        
        except Exception as e:
            raise CustomException(e, sys)

