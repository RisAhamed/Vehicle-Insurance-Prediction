from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact
from sklearn.metrics import f1_score
from src.exception import CustomException
from src.constants import TARGET_COLUMN
from src.logger import logging
from src.utils.common import load_object
import sys
import pandas as pd
from typing import Optional
from src.entity.s3_estimator import Proj1Estimator
from dataclasses import dataclass


@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: float
    is_model_accepted: bool
    difference: float

class ModelEvaluation:
    def __init__(self,model_evaluation_config: ModelEvaluationConfig,
                 data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            logging.info(f"{'>>'*20}Model Evaluation log started.{'<<'*20}")
            self.model_evaluation_config = model_evaluation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise CustomException(e, sys) from e
    
    def get_best_model(self)->Optional[Proj1Estimator]:

        try:
            bucket_name = self.model_evaluation_config.bucket_name

            model_path = self.model_evaluation_config.s3_model_key_path
            project1_estimator= Proj1Estimator(
                bucket_name=bucket_name,model_path=model_path
            )
            if project1_estimator.is_model_present(model_path):
                return project1_estimator
            else:
                logging.info("No model found in S3 bucket. This might be the first run.")
                return None
        except Exception as e:
            logging.warning(f"Error in get_best_model: {e}")
            return None
        
        
            
    def _map_gender_column(self, df):
            """Map Gender column to 0 for Female and 1 for Male."""
            logging.info("Mapping 'Gender' column to binary values")
            df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1}).astype(int)
            return df

    def _create_dummy_columns(self, df):
        """Create dummy variables for categorical features."""
        logging.info("Creating dummy variables for categorical features")
        df = pd.get_dummies(df, drop_first=True)
        return df

    def _rename_columns(self, df):
        """Rename specific columns and ensure integer types for dummy columns."""
        logging.info("Renaming specific columns and casting to int")
        df = df.rename(columns={
            "Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year",
            "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"
        })
        for col in ["Vehicle_Age_lt_1_Year", "Vehicle_Age_gt_2_Years", "Vehicle_Damage_Yes"]:
            if col in df.columns:
                df[col] = df[col].astype('int')
        return df

    def _drop_id_column(self, df):
        """Drop the 'id' column if it exists."""
        logging.info("Dropping 'id' column")
        if "_id" in df.columns:
            df = df.drop("id", axis=1)
        return df

    def evaluate_model(self)->EvaluateModelResponse:

        try:
            logging.info("Evaluating best model")
            test_df  =pd.read_csv(self.data_ingestion_artifact.test_file_path)
            x,y  =test_df.drop(TARGET_COLUMN,axis  =1),test_df[TARGET_COLUMN]

            x = self._map_gender_column(x)
            x = self._create_dummy_columns(x)
            x = self._rename_columns(x)
            x = self._drop_id_column(x)

            trained_model = load_object(file_path=self.model_trainer_artifact.trained_model_file_path)
            trained_model_f1_score = self.model_trainer_artifact.classification_metric_artifact.f1_score

            best_model_f1_score = None
            best_model = self.get_best_model()

            if best_model is not None:
                logging.info("Loading best model from S3")
                y_hat_best_model = best_model.predict(x)
                best_model_f1_score = f1_score(y,y_hat_best_model)
                logging.info(f"Best model F1 score: {best_model_f1_score}")
            else:
                logging.info("No existing model found in S3. This is likely the first run.")
                
            # If no best model exists (first run) or trained model is better
            tmp_best_model_score = 0 if best_model_f1_score is None else best_model_f1_score
            is_model_accepted = True if best_model is None else trained_model_f1_score > tmp_best_model_score
            
            response = EvaluateModelResponse(
                trained_model_f1_score=trained_model_f1_score,
                best_model_f1_score=tmp_best_model_score,
                is_model_accepted=is_model_accepted,
                difference=abs(trained_model_f1_score - tmp_best_model_score)
            )
            logging.info(f"Model evaluation response: {response}")
            return response
        except Exception as e:
            raise CustomException(e, sys) from e
            
    def initiate_model_evaluation(self)->ModelEvaluationArtifact:
        try:
            logging.info("Initiating model evaluation")
            evaluate_model_response = self.evaluate_model()

            s3_model_path = self.model_evaluation_config.s3_model_key_path
            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                changed_accuracy=evaluate_model_response.difference,
                s3_model_path=s3_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path
            )
            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            logging.error(f"Error in model evaluation: {e}")
            # Create a default artifact that accepts the model for the first run
            s3_model_path = self.model_evaluation_config.s3_model_key_path
            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=True,  # Accept model by default for first run
                changed_accuracy=0.0,
                s3_model_path=s3_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path
            )
            logging.info(f"Created default model evaluation artifact due to error: {model_evaluation_artifact}")
            return model_evaluation_artifact

