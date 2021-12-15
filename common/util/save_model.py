"""
save_model.py

Defines save_model
"""

__author__ = "Julien Lefebvre, Hugo Chauvary"
__email__ = 'numerai_2021@protonmail.com'

import joblib

from common.config.constant import DATA_FOLDER


def save_model(model, filename: str) -> None:
    """
    Saves model in joblib format

    args:
        - model: module.model.Model child class model attribute
        - filename: target file name

    """
    joblib.dump(model, f'{DATA_FOLDER}/model/{filename}.joblib')
