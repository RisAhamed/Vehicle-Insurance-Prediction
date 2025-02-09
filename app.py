from dotenv import load_dotenv
load_dotenv()
from src.logger import logging
from src.pipline.training_pipeline import TrainingPipeline
from src.exception import CustomException
import sys

def main():
    try:
        training_pipeline = TrainingPipeline()
        data_ingestion_artifact = training_pipeline.start_data_ingestion()
        data_validation_artifact = training_pipeline.start_data_validation(data_ingestion_artifact)
        
    except Exception as e:
        raise CustomException(e, sys)

if __name__ == "__main__":
    main()
