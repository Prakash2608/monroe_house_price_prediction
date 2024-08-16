import os
import sys
import mlflow
import mlflow.sklearn
import numpy as np
import pickle
import dagshub
import tensorflow

from src.utils.utils import load_object
from urllib.parse import urlparse
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.logger.logging import logging
from src.exceptions.exception import customexception


class ModelEvaluation:
    def __init__(self):
        logging.info("Evaluation started.")
        pass
    
    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        logging.info("Evaluation metrics captured")
        return rmse, mae, r2
    
    
    def initiate_model_evaluation(self, train_array, test_array):
        try:
            X_test, y_test = (test_array[:, :-1], test_array[:,-1])
            
            model_path = os.path.join("artifacts", "model.pkl")
            model = load_object(model_path)
            
            logging.info("model has registered")
            
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            
            
            
        except Exception as e:
            logging.info("Exception occurred during model evaluation")
            raise customexception(e, sys)