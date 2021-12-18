"""
s3.py

Defines S3
"""

__author__ = "Julien Lefebvre, Hugo Chauvary"
__email__ = 'numerai_2021@protonmail.com'

from botocore.exceptions import ClientError

from common.module.aws.client import Client
from common.config.constant import AWS_REGION, DATA_FOLDER, S3_BUCKET


class S3:
    def __init__(self):
        self.client = Client.set_client('s3')


    def upload_file(self,
                    file_name: str,
                    bucket_name: str = S3_BUCKET) -> dict:
        """
        Upload a file to an S3 bucket

        param:
            - file_name: File to upload

        return:
            - formatted boto3 response
        """
        try:
            response = self.client.upload_file(
                f'{DATA_FOLDER}/model/{file_name}.joblib',
                bucket_name,
                f'{file_name}.joblib'
            )

            return response

        except ClientError as e:
            raise e


    def download_file(self,
                      file_name: str,
                      bucket_name: str) -> dict:
        """
        Download file from s3 bucket

        param:
            - file_name: file to download
            - bucket_name: bucket to download from
        """
        try:
            self.client.download_file(bucket_name,
                                      file_name,
                                      f'{DATA_FOLDER}/model/{file_name}')

            return True

        except ClientError as e:
            raise e


    def list_files(self, bucket_name: str = S3_BUCKET) -> list:
        """
        List files in S3 bucket

        args:
            - bucket_name: target bucket
        """
        files = self.client.list_objects_v2(Bucket=bucket_name)['Contents']

        return files
