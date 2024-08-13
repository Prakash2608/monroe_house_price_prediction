import os
import sys
import pickle
import numpy as np
import pandas as pd
from src.logger.logging import logging
from src.exceptions.exception import customexception

from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error
from sklearn.model_selection import ShuffleSplit, cross_val_score, GridSearchCV


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise customexception(e, sys)
    
def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            # Train model
            model.fit(X_train,y_train)

            

            # Predict Testing data
            y_test_pred =model.predict(X_test)

            # Get R2 scores for train and test data
            #train_model_score = r2_score(ytrain,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]] =  test_model_score

        return report

    except Exception as e:
        logging.info('Exception occured during model training')
        raise customexception(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise customexception(e,sys)
    
    
def find_best_model_using_gridsearchcv(X, y, models, scoring, refit_metric):
    try:
        logging.info("Grid search cv training started")
        
        scores = []
        cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
        for algo_name, config in models.items():
            gs =  GridSearchCV(config['model'], config['params'],scoring = scoring, refit=refit_metric, cv=cv, return_train_score=False)
            gs.fit(X,y)
            scores.append({
                'model': algo_name,
                'best_score': gs.best_score_,
                'best_params': gs.best_params_
            })
        logging.info("Grid search cv training completed")
            
        return scores
            
        
    except Exception as e:
        logging.info('Exception occured during grid search cross validation')
        raise customexception(e,sys)