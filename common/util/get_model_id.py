"""
get_model_id.py

Defines get_model_id
"""

__author__ = "Julien Lefebvre, Hugo Chauvary"
__email__ = 'numerai_2021@protonmail.com'

import pandas as pd

from common.config.constant import DATA_FOLDER


def get_model_id(name: str) -> str:
    """
    Read model_id from external file.

    args:
        - name: model name as seen on numerai

    returns:
        - model_id
    """
    df = pd.read_csv(f'{DATA_FOLDER}/credential/model.csv')
    model_id = df[df['name'] == name]['id'].to_string(index=False).strip()

    return model_id
