import sys
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer

from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, CURRENT_YEAR
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.exception import CustomException
from src.logger import logging
from src.utils.common import save_object, save_numpy_array_data, read_yaml

class DataTransformation:
    def __init__(self,
                data_ingestion_artifact: DataIngestionArtifact,
                data_validation_artifact: DataValidationArtifact,
                data_transformation_config: DataTransformationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
            self.schema_config = read_yaml(SCHEMA_FILE_PATH)
        except Exception as e:
            raise CustomException(e,sys)
        
    @staticmethod
    def read_data(file_path: str)-> pd.DataFrame:
        try:
            
            return pd.read_csv(file_path)
        except Exception as e:
            logging.error(f"Error reading data from {file_path}: {e}")
            raise CustomException(e,sys)
        
    def  get_data_transformation_object(self)->Pipeline:
        logging.info(f"Creating data transformation object")
        try:
            numerical_transformer = StandardScaler()
            min_max_scaler =MinMaxScaler()
            num_features = self.schema_config['num_features']
            mm_columns= self.schema_config["mm_columns"]
            logging.info(f"Numerical columns: {num_features}")
            logging.info(f"Categorical columns: {mm_columns}")
            preprocessor = ColumnTransformer(
                transformers=[
                    ("standardScaler",numerical_transformer,num_features),
                    ("minMaxScaler",min_max_scaler,mm_columns)
                ],
                remainder="passthrough"
            )
            final_pipeline =Pipeline(steps=[("preprocessor",preprocessor)])
            logging.info(f"Pipeline created successfully")
            return final_pipeline
        except Exception as e:
            logging.error(f"Exception occurred in get_data_transformer_object method of DataTransformation class: {e}")
            raise CustomException(e,sys)
    # def initiate_data_transformation(self)->DataTransformationArtifact:
    def map_gender_column(self,df: pd.DataFrame):
        logging.info(f"Mapping gender column")
        try:
            df['Gender'] = df['Gender'].map({'Male':1,'Female':0}).astype(int)
            return df
        except Exception as e:
            logging.error(f"Exception occurred in map_gender_column method of DataTransformation class: {e}")
            raise CustomException(e,sys)
        
    def create_dummy_columns(self,df):
        logging.info(f"Creating dummy columns")
        try:
            df = pd.get_dummies(df,drop_first = True)
            logging.info(f"Dummy columns created successfully")
            return df
        except Exception as e:
            logging.error(f"Exception occurred in create_dummy_columns method of DataTransformation class: {e}")
            raise CustomException(e,sys)
    def rename_columns(self,df):
        logging.info(f"Renaming columns")
        
        df = df.rename(columns={
            "Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year",
            "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"
        })
        for col in ["Vehicle_Age_lt_1_Year", "Vehicle_Age_gt_2_Years", "Vehicle_Damage_Yes"]:
            if col in df.columns:
                df[col] = df[col].astype('int')
        return df
    
    def drop_id_column(self,df):
        logging.info(f"Dropping ID column")
        try:
            drop_col =self.schema_config['drop_columns']
            df = df.drop(drop_col,axis=1)
            logging.info(f"ID column dropped successfully")
            return df
        except Exception as e:
            logging.error(f"Exception occurred in drop_id_column method of DataTransformation class: {e}")
            raise CustomException(e,sys)

    def initiate_data_transformation(self)->DataTransformationArtifact:
        logging.info(f"Initiating data transformation")
        try:
            if not self.data_validation_artifact.validation_result_details:
                raise Exception("Validation failed")
            train_df = self.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(file_path=self.data_ingestion_artifact.test_file_path)

            input_feature_train_df = train_df.drop(columns = [TARGET_COLUMN],axis =1)
            target_feature_train_df =train_df[TARGET_COLUMN]

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]


            input_feature_train_df = self.map_gender_column(input_feature_train_df)
            input_feature_train_df =self.drop_id_column(input_feature_train_df)
            input_feature_train_df = self.create_dummy_columns(input_feature_train_df)
            input_feature_train_df = self.rename_columns(input_feature_train_df)

            input_feature_test_df = self.map_gender_column(input_feature_test_df)
            input_feature_test_df =self.drop_id_column(input_feature_test_df)
            input_feature_test_df = self.create_dummy_columns(input_feature_test_df)
            input_feature_test_df = self.rename_columns(input_feature_test_df)

            logging.info(f"Train and test data transformed successfully")
            logging.info(f"Train and test data transformed successfully")
            logging.info("Started data transformation")
            

            preprocessor = self.get_data_transformation_object()
            input_feature_train_df = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_df =preprocessor.transform(input_feature_test_df)

            smt = SMOTEENN(random_state=42,sampling_strategy="minority")
            input_feature_train_df_final,target_feature_train_df_final = smt.fit_resample(input_feature_train_df,target_feature_train_df)
            
            input_feature_test_df_final,target_feature_test_df_final = smt.fit_resample(
                input_feature_test_df,target_feature_test_df
            )

            train_arr = np.c_[input_feature_train_df_final,np.array(target_feature_train_df_final)]
            test_arr = np.c_[input_feature_test_df_final,np.array(target_feature_test_df_final)]

            logging.info(f"Train and test data transformed successfully")
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            logging.info("Saving transformation object and transformed files.")

            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path= self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path= self.data_transformation_config.transformed_test_file_path
            )

            logging.info(f"Data transformation artifact: {data_transformation_artifact}")
            return data_transformation_artifact

           

        except Exception as e:
            raise CustomException(e,sys)

