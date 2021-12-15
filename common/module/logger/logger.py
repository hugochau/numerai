"""
logger.py

Defines a custom logger. Pushed to both console and file
"""

__author__ = "Julien Lefebvre, Hugo Chauvary"
__email__ = 'numerai_2021@protonmail.com'

import logging
import sys

from common.module.logger._formatter import _Formatter
# from module.logger.adapter import Adapter
from common.config.constant import (
    LOG_FILENAME,
    LOG_FILEPATH,
    LOG_LEVEL
)


class Logger():
    def __init__(self):
        """
        Class instantiation
        """
        self.logger = Logger.get_logger()


    @staticmethod
    def get_logger():
        """
        Builds logger object

        returns:
            - logger, 'Logger'
        """
        # init logger
        logger = logging.Logger(LOG_FILENAME)
        logger.setLevel(LOG_LEVEL)

        # enriching logger.record with add. content
        # not being used atm but kept for doc
        # logger.addFilter(Adapter(type))

        # setting up stream logger
        stream_logger = logging.StreamHandler()
        # setting the _Formatter formatter type
        stream_logger.setFormatter(_Formatter)
        stream_logger.setFormatter(_Formatter(
            '%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(message)s'
        ))
        # adding handlers
        logger.addHandler(stream_logger)

        # setting up file logger
        try:
            file_logger = logging.FileHandler(LOG_FILEPATH, 'a+')
        # raise error if file path not found
        except FileNotFoundError as e:
            sys.exit(e)
        file_logger.setFormatter(_Formatter(
            '%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(message)s'
        ))
        logger.addHandler(file_logger)

        return logger