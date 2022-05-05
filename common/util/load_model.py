"""
load_model.py

Defines load_model
"""

__author__ = "Julien Lefebvre, Hugo Chauvary"
__email__ = 'numerai_2021@protonmail.com'

import re

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

    # list bucket files
    bucket = 'numerai-model-staging' if test == '_test' else S3_BUCKET
    files = sss.list_files(bucket)

    # check if there is a corresponding file for designated model
    for file in files:
        # if file['Key'].split('_')[0] == model_name:
        if re.search(f'^{model_name}.*', file['Key']):
            logger.info(f"Downloading {bucket}.{file['Key']}")
            filename = file['Key']
            sss.download_file(file['Key'], bucket, 'model')

            # algo = file['Key'].split('_')[1].split('.')[0]
            # algo = algo[0].upper() + algo[1:]

            # model = eval(algo)(None, None, True, file['Key'])

    # return model
    return filename
