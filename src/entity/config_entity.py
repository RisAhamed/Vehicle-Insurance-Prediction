import os
from src.constants import *
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

@dataclass
class TrainingPipelineConfig:
    pipeline_name : str = PIPELINE_NAME
    artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)
    timestamp: str = TIMESTAMP

training_pipeline_config = TrainingPipelineConfig()

@dataclass
class DataIngestionConfig:
    root_dir: Path=  os.path.join(training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME)
    source_URL: str = DATA_INGESTION_URL
    local_data_file: Path = os.path.join(root_dir, "data.csv")
    unzip_dir: str = os.path.join(root_dir, "unzipped_data")
    raw_data_dir: str = os.path.join(root_dir, "raw_data")
    ingested_data_dir: str = os.path.join(root_dir, "ingested_data")

    train_file_path: str = os.path.join(ingested_data_dir, "train", TRAIN_FILE_NAME)
    test_file_path: str = os.path.join(ingested_data_dir, "test", TEST_FILE_NAME)
    train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO

@dataclass
class DataValidationConfig:
    data_validation_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_VALIDATION_DIR_NAME)
    validation_report_file_path: str = os.path.join(data_validation_dir, DATA_VALIDATION_REPORT_FILE_NAME)

# @dataclass
# class DataIngestionConfig:
#     d
#     feature_store_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR, FILE_NAME)
#     
#     collection_name: str = DATA_INGESTION_COLLECTION_NAME
#     database_name: str = DATABASE_NAME

# data_ingestion_config = DataIngestionConfig()

