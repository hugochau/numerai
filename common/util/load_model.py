"""
load_model.py

Defines load_model
"""

__author__ = "Julien Lefebvre, Hugo Chauvary"
__email__ = 'numerai_2021@protonmail.com'

from common.module.aws import S3
# from common.module.model import *
from common.module.logger import Logger


def load_model(model_name: str):
    """
    Downloads model from s3 and load underlying module

    args:
        - model_name

    returns:
        - model: module.model.Model child class instance
    """
    sss = S3()
    logger = Logger().logger

    # list bucket files
    files = sss.list_files()

    # check if there is a corresponding file for designated model
    for file in files:
        if file['Key'].split('_')[0] == model_name:
            logger.info(f"Downloading {file['Key']}")
            sss.download_file(file['Key'], 'numerai-model')

            # algo = file['Key'].split('_')[1].split('.')[0]
            # algo = algo[0].upper() + algo[1:]

            # model = eval(algo)(None, None, True, file['Key'])

    # return model
