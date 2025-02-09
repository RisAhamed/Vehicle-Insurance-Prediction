from dotenv import load_dotenv
load_dotenv()

from src.logger import logging
from src.pipline.training_pipeline import TrainingPipeline
from src.exception import CustomException
import sys

from src.config.configuration import ConfigurationManager
from src.components.data_ingestion import DataIngestion

def main():
    config = ConfigurationManager()
    data_ingestion_config = config.get_data_ingestion_config()
    data_ingestion = DataIngestion(config=data_ingestion_config)
    data_ingestion.initiate_data_ingestion()



if __name__ == "__main__":
    main()
