from dataclasses import dataclass

@dataclass 
class DataIngestionArtifact:
    trained_file_path: str
    test_file_path : str
    
@dataclass
class DataValidationArtifact:
    validation_status: bool
    message: str
    validation_result_details: str
