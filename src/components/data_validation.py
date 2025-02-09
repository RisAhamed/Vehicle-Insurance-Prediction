import json
import pandas as pd
import numpy as np
from src.entity.config_entity import *
from src.entity.artifact_entity import *
from src.utils.common import read_yaml, write_yaml_file
from src.logger import logging
from src.exception import CustomException
import sys



class DataValidation:
    def __init__(self, data_validation_config: DataValidationConfig, data_ingestion_artifact: DataIngestionArtifact):
        try:
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.schema_config = read_yaml(SCHEMA_FILE_PATH)

        except Exception as e:
            raise CustomException(e, sys) from e
    def validate_number_of_columns(self,df: pd.DataFrame)->bool:
         try:
              number_of_columns = len(df.columns)
              status=number_of_columns == self.schema_config["columns"]
              logging.info(f"number of columns: {number_of_columns}")
              logging.info(f"status: {status}")
              return status
         except Exception as e:
              raise CustomException(e, sys) from e
         
    def is_column_exists(self,df: pd.DataFrame,)->bool:
    
        try:
            dataframe_columns = df.columns
            missing_numerical_columns = []
            missing_categorical_columns = []
            for column in self.schema_config["numerical_columns"]:
                if column not in dataframe_columns:
                    missing_numerical_columns.append(column)


            if len(missing_numerical_columns)>0:
                logging.info(f"Missing numerical column: {missing_numerical_columns}")


            for column in self.schema_config["categorical_columns"]:
                if column not in dataframe_columns:
                    missing_categorical_columns.append(column)


            if len(missing_categorical_columns)>0:
                logging.info(f"Missing categorical column: {missing_categorical_columns}")

            return False if len(missing_categorical_columns)>0 or len(missing_numerical_columns)>0 else True
        except Exception as e:
              raise CustomException(e, sys) from e

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e, sys) from e
        
    def initiate_data_validation(self)->DataValidationArtifact:
        try:
            validation_error_message=""
            logging.info(f"Starting data validation")
            train_df = DataValidation.read_data(self.data_ingestion_artifact.trained_file_path)

            test_df = DataValidation.read_data(self.data_ingestion_artifact.test_file_path)
            validation_status = self.validate_number_of_columns(train_df)
            if not validation_status:
                validation_error_message+=f"Train data is not having required number of columns\n"
            validation_status = self.validate_number_of_columns(test_df)
            if not validation_status:
                validation_error_message+=f"Test data is not having required number of columns\n"
            validation_status = self.is_column_exists(train_df)
            if not validation_status:
                validation_error_message+=f"Train data is not having required columns\n"
            validation_status = self.is_column_exists(test_df)
            if not validation_status:
                validation_error_message+=f"Test data is not having required columns\n"
            if validation_error_message=="":
                validation_status=True
                logging.info(f"Data validation completed successfully")
            
            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=validation_error_message,
                validation_result_details=self.data_validation_config.validation_report_file_path
            )
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            report_dir = os.path.dirname(self.data_validation_config.validation_report_file_path)
            os.makedirs(report_dir, exist_ok=True)

            # Save validation status and message to a JSON file
            validation_report = {
                "validation_status": validation_status,
                "message": validation_error_message.strip()
            }

            with open(self.data_validation_config.validation_report_file_path, "w") as report_file:
                json.dump(validation_report, report_file, indent=4)

            logging.info("Data validation artifact created and saved to JSON file.")
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
            
        except Exception as e:
            raise CustomException(e, sys) from e
            
        