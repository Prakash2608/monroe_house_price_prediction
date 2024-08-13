import pandas as  pd
import numpy as np
from src.logger.logging import logging
from src.exceptions.exception import customexception
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from src.utils.utils import save_object, load_object, find_best_model_using_gridsearchcv

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor


@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')
    
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
        
    def initiate_model_training(self,train_df,test_df):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            
            train_array = np.array(train_df)
            test_array = np.array(test_df)
            
            
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            models = {
                'linear_regression' : {
                    'model': LinearRegression(),
                    'params': {
                        'fit_intercept': [True, False]
                    }
                },
                'lasso': {
                    'model': Lasso(),
                    'params': {
                        'alpha': [1,2],
                        'selection': ['random', 'cyclic']
                    }
                },
                'decision_tree': {
                    'model': DecisionTreeRegressor(),
                    'params': {
                        'criterion' : ['absolute_error','friedman_mse'],
                        'splitter': ['best','random']
                    }
                },
                'Ridge':{
                    'model':Ridge(),
                    'params':{
                        'fit_intercept' : [True, False],
                        'alpha' : [0.1, 0.05, 0.2] 
                    }
                },
                'elastic_net':{
                    'model': ElasticNet(),
                    'params':{
                        'alpha': [1,2],
                        'l1_ratio': [0.1, 0.5, 0.9]
                    }
                }
            }
            scoring = ['neg_mean_squared_error', 'neg_median_absolute_error', 'r2']
            refit_metric = 'r2'
            
            report = find_best_model_using_gridsearchcv(np.concatenate((X_train,X_test), axis =0), np.concatenate((y_train,y_test), axis =0),
                                                        models, scoring, refit_metric )
            
            score = {result['model']: result['best_score'] for result in report}
                    
            print(score)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {score}')
            
            # To get best model score from dictionary 
            best_model_score = max(score.values())
            best_model_name = [key for key, value in score.items() if value == best_model_score][0]
            best_params = [result['best_params']  for result in report if result['model']==best_model_name]

            # best_model_name = list(score.keys())[
            #     list(score.values()).index(best_model_score)
            # ]
            
            best_model = models[best_model_name]['model']
            
            best_model.set_params(**best_params[0])
            best_model.fit(np.concatenate((X_train, X_test), axis=0), np.concatenate((y_train, y_test), axis=0))
            
            logging.info(f'Best Model: {best_model_name} with a score of {best_model_score}')
            
            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model
            )
            
            # best_model = models[best_model_name]['model']
            # best_params = models[best_model_name]['params']
            
        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise customexception(e,sys)