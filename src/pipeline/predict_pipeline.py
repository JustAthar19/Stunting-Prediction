'''
This script aims to create the predict pipeline for a simple web application which will be interacting with the pkl files, such that we can make predictions by giving values of input features. 
'''

# Debugging and verbose.
import sys
from src.logger import logging
from src.exception import CustomException
from src.utils import load_object

# Data manipulation.
import pandas as pd

# File handling.
import os


class PredictPipeline:
    '''
    Class for making predictions using a trained model and preprocessor.

    This class provides a pipeline for making predictions on new instances using a trained machine learning model and
    a preprocessor. It loads the model and preprocessor from files, preprocesses the input features, and makes predictions.

    Methods:
        predict(features):
            Make predictions on new instances using the loaded model and preprocessor.

    Example:
        pipeline = PredictPipeline()
        new_features = [...]
        prediction = pipeline.predict(new_features)

    Note:
        This class assumes the availability of the load_object function.
    '''

    
    def __init__(self) -> None:
        '''
        Initializes a PredictPipeline instance.

        Initializes the instance. No specific setup is required in the constructor.
        '''
        pass


    def predict(self, features):
        '''
        Make predictions on new instances using the loaded model and preprocessor.

        Args:
            features: Input features for which predictions will be made.

        Returns:
            predictions: Predicted labels for the input features.

        Raises:
            CustomException: If an exception occurs during the prediction process.
        '''
        try:


            classes_dict = {
                0:"normal",
                1:"Stunting Tinggi",
                2:"stunting",
                3:"gizi baik"
            }

            model_path = os.path.join('models', 'lightGBM.pkl')
            
            logging.info('Load model objects.')

            model = load_object(file_path=model_path)

            logging.info('Preprocess the input data.')
            
            # Assert input data is float64 data type.
            prepared_data = features.astype('float64')
            
            logging.info('Predict.')
            
            # Predict customer's churn probability.
            predicted_proba = model.predict_proba(prepared_data,predict_disable_shape_check=True)[:, 1][0]
            class_prediction = model.predict(prepared_data)

            # Prediction output (customer's probability of churning).
            prediction = f"""Probabilitas anak terkena stunting: {round(predicted_proba * 100, 3)}%
                        Diagnosa akhir: {classes_dict[class_prediction[0]]}"""

            logging.info('Prediction successfully made.')

            return prediction

        except Exception as e:
            raise CustomException(e, sys)
        

class InputData:
    '''
    Class for handling input data for predictions.

    This class provides a structured representation for input data that is meant to be used for making predictions.
    It maps input variables from HTML inputs to class attributes and provides a method to convert the input data into
    a DataFrame format suitable for making predictions.

    Attributes:

    Methods:
        get_input_data_df():
            Convert the mapped input data into a DataFrame for predictions.

    Note:
        This class assumes the availability of the pandas library and defines the CustomException class.
    '''

    def __init__(self,
                 umur: int,
                 jenis_kelamin:int,
                 tinggi_badan:float
                 ) -> None:
        '''
        Initialize an InputData instance with mapped input data.

        Args:
            umur (int)              
            lingkar_kepala (float)
            lingkar_lengan (float)

        '''
        
        # Map variables from html inputs.
        self.umur = umur
        self.jenis_kelamin = jenis_kelamin
        self.tinggi_badan = tinggi_badan

    def get_input_data_df(self):
        '''
        Convert the mapped input data into a DataFrame for predictions.

        Returns:
            input_data_df (DataFrame): DataFrame containing the mapped input data.

        Raises:
            CustomException: If an exception occurs during the process.
        '''
        try:
            input_data_dict = dict()

            # Map the variables to the form of a dataframe for being used in predictions.
            
            input_data_dict['umur'] = [self.umur]
            input_data_dict['jenis_kelamin'] = [self.jenis_kelamin]
            input_data_dict['tinggi_badan'] = [self.tinggi_badan]

            input_data_df = pd.DataFrame(input_data_dict)

            return input_data_df
        
        except Exception as e:
            raise CustomException(e, sys)