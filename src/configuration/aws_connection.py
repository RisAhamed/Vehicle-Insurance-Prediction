import os
import boto3
from src.constants import AWS_ACCESS_KEY_ID_ENV_KEY, AWS_SECRET_ACCESS_KEY_ENV_KEY, REGION_NAME
from src.exception import CustomException
import sys

class S3Connection:
    def __init__(self):
        try:
            self.access_key_id = os.getenv(AWS_ACCESS_KEY_ID_ENV_KEY)
            self.secret_access_key = os.getenv(AWS_SECRET_ACCESS_KEY_ENV_KEY)
            
            if not self.access_key_id or not self.secret_access_key:
                raise CustomException("AWS credentials not found in environment variables")

            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=self.access_key_id,
                aws_secret_access_key=self.secret_access_key,
                region_name=REGION_NAME
            )

            self.s3_resource = boto3.resource(
                's3',
                aws_access_key_id=self.access_key_id,
                aws_secret_access_key=self.secret_access_key,
                region_name=REGION_NAME
            )
            
        except Exception as e:
            raise CustomException(e, sys)
