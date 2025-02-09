from src.constants  import *
from src.entity.config_entity import *
from src.utils.common import *
from box import Box
import yaml

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):
        
        self.config = self.load_config()
        # self.params = read_yaml(params_filepath)
        
        create_directories([self.config.artifacts_root])

    def load_config(self):
        with open("config/config.yaml", "r") as file:
            config = yaml.safe_load(file)
            return Box(config)  # Convert to Box for easier access

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )

        return data_ingestion_config 
    
    def get_data_validation_config(self):
        return self.config.data_validation  # Ensure this matches your YAML structure

