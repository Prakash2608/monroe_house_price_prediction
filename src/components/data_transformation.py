import pandas as pd
import numpy as np
from src.logger.logging import logging
from src.exceptions.exception import customexception
import os
import sys
from dataclasses import dataclass
from pathlib import Path


from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

from src.utils.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path:str = os.path.join('artifacts','preprocessor.pkl')
    train_data_path:str = os.path.join("artifacts","train.csv")
    test_data_path:str = os.path.join("artifacts","test.csv")
    

class DropUnwantedColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns_to_drop)
    
class DropNullValues(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.dropna()
    
class FeatureEngineeringBHK(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['bhk'] = X['size'].apply(lambda x: int(x.split(' ')[0]))
        return X
    

class ConvertSqftToNum(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def convert_sqft_to_num(x):
            tokens = x.split('-')
            if len(tokens) == 2:
                return (float(tokens[0]) + float(tokens[1])) / 2
            try:
                return float(x)
            except:
                return None
        X['total_sqft'] = X['total_sqft'].apply(convert_sqft_to_num)
        X = X[X['total_sqft'].notnull()]
        return X
    
    
class FeatureEngineeringPricePerSqft(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['price_per_sqft'] = X['price'] * 100000 / X['total_sqft']
        return X
    

class FeatureEngineeringLocation(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['location'] = X['location'].apply(lambda x: x.strip())
        location_stats = X['location'].value_counts(ascending=False)
        location_stats_less_than_10 = location_stats[location_stats <= 10]
        X['location'] = X['location'].apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
        return X
    

class OutlierRemovalPricepersqft(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X[~(X.total_sqft/X.bhk<300)]
        df_out = pd.DataFrame()
        
        for key, subdf in X.groupby('location'):
            m = np.mean(subdf.price_per_sqft)
            st = np.std(subdf.price_per_sqft)
            reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
            df_out = pd.concat([df_out,reduced_df],ignore_index=True)
            
        return df_out
    
    
class OutlierRemovalBHK(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        exclude_indices = np.array([])
        for location, location_df in X.groupby('location'):
            bhk_stats = {}
            for bhk, bhk_df in location_df.groupby('bhk'):
                bhk_stats[bhk] = {
                    'mean': np.mean(bhk_df.price_per_sqft),
                    'std': np.std(bhk_df.price_per_sqft),
                    'count': bhk_df.shape[0]
                }
            for bhk, bhk_df in location_df.groupby('bhk'):
                stats = bhk_stats.get(bhk-1)
                if stats and stats['count']>5:
                    exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
                    
        return X.drop(exclude_indices,axis='index')
    
    

class OutlierRemovalBathroom(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X[X.bath<X.bhk+2]
        X = X.drop(['size','price_per_sqft'],axis='columns')
        
        return X
    

class LocationOneHotEncoding(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        location_dummies = pd.get_dummies(X.location)
        X = pd.concat([X, location_dummies.drop('other', axis = 'columns')], axis='columns')
        X = X.drop('location',axis='columns')
        
        return X
    


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        
    def get_data_transformation(self):
        try:
            logging.info('Data transformation pipeline initiated')
            
            pipeline = Pipeline(steps=[
                ('drop_unwanted_columns', DropUnwantedColumns(columns_to_drop=['area_type', 'society', 'balcony', 'availability'])),
                ('drop_null_values', DropNullValues()),
                ('feature_engineering_bhk', FeatureEngineeringBHK()),
                ('convert_sqft_to_num', ConvertSqftToNum()),
                ('feature_engineering_price_per_sqft', FeatureEngineeringPricePerSqft()),
                ('feature_engineering_location', FeatureEngineeringLocation()),
                ('outlier_removal_pricepersqft', OutlierRemovalPricepersqft()),
                ('outlier_removal_bhk', OutlierRemovalBHK()),
                ('outlier_removal_bathroom', OutlierRemovalBathroom()),
                ('location_one_hot_encoding', LocationOneHotEncoding())
            ])
            
            return pipeline
            
        except Exception as e:
            logging.info('Error in data transformation pipeline')
            raise customexception(e, sys)
                    
                
                
    def initialize_data_transformation(self,file_path):
        try:
            df = pd.read_csv(file_path)
            
            logging.info("read and load file is completed")
            
            preprocessing_obj = self.get_data_transformation()
            
            preprocessed_df = preprocessing_obj.fit_transform(df)
            
            logging.info("Preprocessing the data is completed")
            target_column_name = 'price'
            
            X = preprocessed_df.drop(target_column_name, axis='columns')
            y = preprocessed_df[target_column_name]
            
            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)
            
            logging.info("DataFrame is splitted into train and test")
            
            train_df = pd.concat((X_train, y_train), axis = 1)
            train_df.reset_index(drop=True, inplace=True)
            test_df = pd.concat((X_test, y_test), axis = 1)
            test_df.reset_index(drop=True, inplace=True)
            
            train_df.to_csv(self.data_transformation_config.train_data_path)
            test_df.to_csv(self.data_transformation_config.test_data_path)
            
            logging.info("train and test data saved to path")
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            return (
                train_df, 
                test_df
            )    
            
            
        except Exception as e:
            logging.info('Error in initializing data transformation')
            raise customexception(e, sys)
            
            