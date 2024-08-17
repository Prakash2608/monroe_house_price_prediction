import os
import sys
from src.logger.logging import logging
from src.exceptions.exception import customexception
import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTrainer
from src.components.model_evaluation import ModelEvaluation


data_ingestion = DataIngestion()

raw_data_path = data_ingestion.initiate_data_ingestion()

data_transformation = DataTransformation()

train_df, test_df = data_transformation.initialize_data_transformation(raw_data_path)

model_trainer_obj = ModelTrainer()
model_trainer_obj.initiate_model_training(train_df, test_df)

model_eval_obj = ModelEvaluation()
model_eval_obj.initiate_model_evaluation(train_df,test_df)