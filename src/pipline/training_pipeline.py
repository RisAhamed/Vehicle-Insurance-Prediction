import sys
import os
from src.components.data_ingestion import DataIngestion
from src.entity.config_entity import TrainingPipelineConfig,DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact

from src.exception import CustomException
from src.logger import logging

class TrainingPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
    
    def start_data_ingestion(self)->DataIngestionArtifact:
        try:
            return self.data_ingestion.initiate_data_ingestion()
        
        except Exception as e:
            raise CustomException(e,sys)
        
    

