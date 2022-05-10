"""
api.py

Implements API
"""

__author__ = "Julien Lefebvre, Hugo Chauvary"
__email__ = 'numerai_2021@protonmail.com'

import os

import numerapi

from common.config.constant import (
    DATA_FOLDER,
    SECRET_KEY,
    PUBLIC_ID,
    DFILES
)
from common.util.log_item import log_item
from common.util.get_model_id import get_model_id


class Api:
    def __init__(self):
        self.api = Api.create_api()


    @staticmethod
    def create_api():
        """
        Create NumerAPI object

        return:
            - NumerAPI class instance
        """
        napi = numerapi.NumerAPI(secret_key=SECRET_KEY,
                                 public_id=PUBLIC_ID)

        return napi


    def download_dataset(self, version: str, type: str) -> None:
        """
        Download contest data into data folder

        args:
            - replace: replace any existing file
                - default to False
        """
        for dfile in DFILES[version][type]:
            dest_path = f"{DATA_FOLDER}/numerai/{version}/{dfile}"
            self.api.download_dataset(f'{version}/{dfile}',
                                      dest_path=dest_path)


    def download_new_dataset(self, type: str) -> None:
        """
        Download new dataset

        args:
            - type: file type
                - either tournament or training
        """
        for dfile in DFILES[type]:
            self.api.download_dataset(dfile,
                                      dest_path=f"{DATA_FOLDER}/numerai/{dfile}")


    def get_competition_round(self) -> float:
        """
        Get latest competition round

        returns:
            - rno: round number
        """
        competitions = self.api.get_competitions()
        rno = competitions[0]['number']

        return rno


    # @log_item
    def upload_predictions(self, model_name: str, data_type: str) -> None:
        """
        Upload predictions

        args:
            - model_name: model name as seen on numerai
            - data_type: configures upload_predictions.version attribute
        """
        version = {
            'v2': 1,
            'v3': 2,
            'v4': 2
        }

        self.api.upload_predictions(f"{DATA_FOLDER}/predictions.csv",
                                    model_id=get_model_id(model_name),
                                    version=(version[data_type] or 2))


    # @log_item
    def upload_diagnostics(self, model_name: str) -> None:
        """
        Upload predictions

        args:
            - model_name: model name as seen on numerai
            - data_type: configures upload_predictions.version attribute
        """

        self.api.upload_diagnostics(f"{DATA_FOLDER}/predictions.csv",
                                    model_id=get_model_id(model_name))
