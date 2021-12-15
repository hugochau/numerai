"""
model.py

Interface for child classes
"""

__author__ = "Julien Lefebvre, Hugo Chauvary"
__email__ = 'numerai_2021@protonmail.com'

from abc import ABC, abstractmethod

import numpy as np


class Model(ABC):
    def __init__(self,
                 X: np.array,
                 y: np.array,
                 pre_trained: bool = False,
                 filename: str = None):
        """
        args:
            - X: training features
            - y: training targets
            - pre_trained: set to True when loading model from joblib file
            - filename: path to joblife file
        """
        pass


    @abstractmethod
    def fit(self,
            X: np.array,
            y: np.array,
            pre_trained: bool,
            filename: str):
        """
        Fit model

        args:
            - X: training features
            - y: training targets
            - pre_trained: set to True when loading model from joblib file
            - filename: path to joblife file
        """
        pass


    @abstractmethod
    def predict(self, X_pred: np.array):
        """
        Build model predictions

        args:
            - X_pred: prediction features
        """
        pass


    @abstractmethod
    def score(self, X_valid: np.array, y_valid: np.array) -> float:
        """
        Compute model accuracy

        args:
            - X_valid: validation features
            - y_valid: validation targets

        returns:
            - score: model accuracy
        """
        pass
