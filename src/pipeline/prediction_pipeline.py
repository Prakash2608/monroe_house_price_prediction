import os
import sys
import pandas as pd
import numpy as np
import json

from src.exceptions.exception import customexception
from src.logger.logging import logging
from src.utils.utils import load_object



class PredictPipeline:
    
    def __init__(self):
        pass
    
    def transform_predict_data(self, features):  
        with open('./columns.json','r') as file:
            column_list = json.load(file)['data_columns']
        
        logging.info("json loaded")
        
        location = features['location'][0]
        total_sqft = features['total_sqft'][0]
        bath = features['bath'][0]
        bhk = features['bhk'][0]
        
        logging.info("data loaded into variables")
        
        loc_index = column_list.index(location)
        
        logging.info("found loc index of location")

        x = np.zeros(len(column_list))
        x[0] = total_sqft
        x[1] = bath
        x[2] = bhk
        if loc_index >= 0:
            x[loc_index] = 1
            
        logging.info(" X converted")

        return x.reshape(1, -1)
    
    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            
            logging.info("data came for prediction")
            
            X = self.transform_predict_data(features)
            
            model = load_object(model_path)
            logging.info("data going for prediction")
            
            logging.info("data preprocessed for prediction")
            
            pred = model.predict(X)
            
            return pred        
        
        except customexception as e:
            logging.info('Exception occurred during prediction')
            raise customexception(e,sys)
        
        
        
        
class CustomData:
    def __init__(self,
                 location:str,
                 total_sqft:float,
                 bath:float,
                 bhk:int):
        self.location = location
        self.total_sqft = total_sqft
        self.bath = bath
        self.bhk = bhk
        
        
    def get_data_as_dataframe(self):
        try:
            data = {'location': [self.location],
                    'total_sqft': [self.total_sqft],
                    'bath': [self.bath],
                    'bhk': [self.bhk]}
            
            df = pd.DataFrame(data)
            logging.info("DataFrame created for prediction")
            
            return df
            
        except customexception as e:
            logging.info('Exception occurred during data conversion to dataframe')
            raise customexception(e,sys)
        