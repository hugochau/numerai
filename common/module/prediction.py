"""
prediction.py

Implements Prediction
"""

__author__ = "Julien Lefebvre, Hugo Chauvary"
__email__ = 'numerai_2021@protonmail.com'

import pandas as pd
import numpy as np

from common.config.constant import DATA_FOLDER

class Prediction:
    def __init__(self, ids: np.array, yhat: np.array) -> None:
        self.ids = ids
        self.yhat = yhat


    def save(self):
        """
        Saves model predictions to csv file
        File follows the requested format by Numerai
        """
        # build subsequent dataframe
        df = pd.DataFrame(data=self.yhat,
                          index=self.ids,
                          columns=['prediction'])
        df.index.rename('id', inplace=True)

        # save predictions to csv file
        df.to_csv(f'{DATA_FOLDER}/predictions.csv')
