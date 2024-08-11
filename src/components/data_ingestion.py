import pandas as pd
import numpy as np
from src.logger.logging import logging
from src.exceptions.exception import customexception
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from sklearn.model_selection import train_test_split


@dataclass
class DataIngestionConfig:
    raw_data_path:str = os.path.join("artifacts","raw.csv")
    
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
        
    def initiate_data_ingestion(self):
        logging.info("Data ingestion started.")
        try:
            data = pd.read_csv("C:/Users/praka/my_personal_project/monroe_house_price_prediction/data/bengaluru_house_prices.csv")
            logging.info("reading a df")
            
            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_data_path)))
            data.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("I have saved the raw dataset in artifact folder.")
            
            # logging.info("Here I have performed train test split")
            
            # train_data, test_data = train_test_split(data, test_size=0.25)
            # logging.info("Train test split completed.")
            
            # train_data.to_csv(self.ingestion_config.train_data_path, index=False)
            # test_data.to_csv(self.ingestion_config.test_data_path, index=False)
            
            logging.info("Data ingestion part completed.")
            
            return (
                self.ingestion_config.raw_data_path
            )
            
        except Exception as e:
            logging.info("Exception occured in data ingestion part")
            raise customexception(e, sys)
        

if __name__ == "__main__":
    obj = DataIngestion()
    
    obj.initiate_data_ingestion()