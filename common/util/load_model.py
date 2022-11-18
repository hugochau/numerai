"""
load_model.py

Defines load_model
"""

__author__ = "Julien Lefebvre, Hugo Chauvary"
__email__ = 'numerai_2021@protonmail.com'

from common.module.aws import S3
from common.module.logger import Logger
from common.config.constant import S3_BUCKET


def load_model(model_name: str, test: str):
    """
    Downloads model from s3 and load underlying module

    args:
        - model_name
        - test: load staging model

    returns:
        - model: module.model.Model child class instance
    """
    sss = S3()
    logger = Logger().logger

    bucket = 'numerai-model-staging' if test == '_test' else S3_BUCKET
    filename = f'{model_name}.joblib'

    logger.info(f"Downloading {bucket}/{filename}")
    sss.download_file(filename, bucket, 'model')

    return filename
