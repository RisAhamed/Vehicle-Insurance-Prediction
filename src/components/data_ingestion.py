import os, sys
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.exception import CustomException
from src.logger import logging
from src.configuration.mongo_db_connection import MongoDBClient
from pandas import DataFrame
from typing import Optional
from sklearn.model_selection import train_test_split
from src.data_access.proj1_data import Proj1Data
from pathlib import Path
import urllib.request as request
import pandas as pd
from src.utils.common import get_size

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            os.makedirs(os.path.dirname(self.config.local_data_file), exist_ok=True)
            filename, headers = request.urlretrieve(
                url = self.config.source_URL,
                filename = self.config.local_data_file
            )

    def extract_data(self):
        """
        Read CSV data and perform basic validation
        """
        os.makedirs(self.config.unzip_dir, exist_ok=True)
        df = pd.read_csv(self.config.local_data_file)
        
        # Basic validation
        assert len(df.columns) == 12, "CSV should have 12 columns"
        required_columns = ["id","Gender","Age","Driving_License","Region_Code","Previously_Insured","Vehicle_Age","Vehicle_Damage","Annual_Premium","Policy_Sales_Channel","Vintage","Response"]
        for col in required_columns:
            assert col in df.columns, f"Missing required column: {col}"

            
        # Save processed data
        df.to_csv(os.path.join(self.config.unzip_dir, "visa_data.csv"), index=False)

    def export_data_into_feature_store(self)->DataFrame:
        """
        Method Name :   export_data_into_feature_store
        Description :   This method exports data from mongodb to csv file
        
        Output      :   data is returned as artifact of data ingestion components
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info(f"Exporting data from mongodb")
            my_data = Proj1Data()
            dataframe = my_data.export_collection_as_dataframe(collection_name=
                                                                   self.config.collection_name,
                                                                   database_name=self.config.database_name)
            logging.info(f"Shape of dataframe: {dataframe.shape}")
            feature_store_file_path  = self.config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)
            logging.info(f"Saving exported data into feature store file path: {feature_store_file_path}")
            dataframe.to_csv(feature_store_file_path,index=False,header=True)
            return dataframe

        except Exception as e:
            raise CustomException(e,sys)
    
    
    
    def split_data_as_train_test_(self,dataframe: DataFrame)->None:
        logging.info(f"Splitting data into train and test")
        try:
            train_set,test_set = train_test_split(dataframe,test_size = self.config.train_test_split_ratio)

            path = Path(self.config.train_file_path)
            dir_path =os.path.dirname(path)
            os.makedirs(dir_path,exist_ok=True)
            logging.info(f"Saving train set into file path: {self.config.train_file_path}")
            train_set.to_csv(self.config.train_file_path,index=False,header=True)
            test_set.to_csv(self.config.test_file_path,index=False,header=True)
            logging.info(f"Train test split completed")

        except Exception as e:
            raise CustomException(e,sys)
        
    
    def initiate_data_ingestion(self)->DataIngestionArtifact:
        try:
            logging.info("Starting data ingestion")
            
            # Create all required directories
            os.makedirs(self.config.root_dir, exist_ok=True)
            os.makedirs(self.config.raw_data_dir, exist_ok=True)
            os.makedirs(os.path.dirname(self.config.train_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.config.test_file_path), exist_ok=True)
            
            logging.info("Downloading and extracting data")
            # Download and extract data from source
            self.download_file()
            self.extract_data()
            
            # Read the extracted data
            logging.info("Reading extracted data")
            df = pd.read_csv(os.path.join(self.config.unzip_dir, "visa_data.csv"))
            
            # Save raw data
            raw_data_path = os.path.join(self.config.raw_data_dir, "visa_data.csv")
            logging.info(f"Saving raw data to {raw_data_path}")
            df.to_csv(raw_data_path, index=False)
            
            # Split data into train and test sets
            logging.info("Splitting data into train and test sets")
            train_set, test_set = train_test_split(
                df, 
                test_size=self.config.train_test_split_ratio,
                random_state=42
            )
            
            # Save train data
            logging.info(f"Saving train data to {self.config.train_file_path}")
            train_set.to_csv(self.config.train_file_path, index=False, header=True)
            
            # Save test data
            logging.info(f"Saving test data to {self.config.test_file_path}")
            test_set.to_csv(self.config.test_file_path, index=False, header=True)
            
            # Return artifact with paths
            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.config.train_file_path,
                test_file_path=self.config.test_file_path
            )
            
            logging.info(f"Data ingestion completed. Artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        
        except Exception as e:
            raise CustomException(e,sys)
            

