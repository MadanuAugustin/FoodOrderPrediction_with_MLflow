
import sys
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from src.FoodOrderPrediction.logger_file.logger_obj import logger
from src.FoodOrderPrediction.Exception.custom_exception import CustomException
from src.FoodOrderPrediction.pipeline.prediction_pipeline import PredictionPipeline



# initializing the flask app

app = Flask(__name__)


# route to display the home page

@app.route('/predict', methods = ['POST', 'GET'])
def predict_datapoint():

    if request.method == 'GET':
        return render_template('index.html')
    

    else : 
        try:
            Age = request.form.get('Age')
            monthly_income = request.form.get('monthly_income')
            family_size = request.form.get('family_size')
            pincode = request.form.get('pincode')
            Gender = request.form.get('Gender')
            Marital_status = request.form.get('Marital_status')
            occupation = request.form.get('occupation')
            feedback = request.form.get('feedback')
            educational_qualifications = request.form.get('educational_qualifications')



            data = [Age, monthly_income, family_size , pincode, Gender , Marital_status, occupation, feedback, educational_qualifications]
            
            logger.info(f'-----------Feteched data successfully from the user--------------')
            

            data = np.array(data).reshape(1, 9)

            data = pd.DataFrame(data)

            print(data)

            obj = PredictionPipeline()

            results = obj.predictDatapoint(data)

            logger.info(f'-----------Below is the final result {results}------------------')

            print(results)

            return render_template('index.html', results = str(results))


        except Exception as e:
            raise CustomException(e, sys)
        



if __name__ == '__main__':
    app.run(host = "0.0.0.0", debug = True) ## http://127.0.0.1:5000

