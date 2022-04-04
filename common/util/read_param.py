"""
read_param.py

Implements Splitter
"""

__author__ = "Julien Lefebvre, Hugo Chauvary"
__email__ = 'numerai_2021@protonmail.com'

import json

from numpy import log

from common.module.aws import S3
from common.module.logger import Logger
from common.config.constant import DATA_FOLDER, S3_BUCKET
from common.module.logger import Logger

def read_param(model_name: str):
    logger = Logger().logger
    logger.info(f"Selected param: {model_name}.json")
    # model_name = model_name[0].lower() + model_name[1:]
    filename = f'{DATA_FOLDER}/model/param/{model_name}.json'

    try:
        with open(filename) as file:
            dict = json.load(file)
    except:
        logger = Logger().logger
        logger.info(f"Could not find parameter file. Loading from S3 instead")
        S3().download_file(f'param/{model_name}.json', S3_BUCKET, 'model')

        with open(filename) as file:
            dict = json.load(file)

    return dict
