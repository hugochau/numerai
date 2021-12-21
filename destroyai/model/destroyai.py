"""
catboost_classifier.py

Implements CatBoost Classifier
Refer to model.model.py for documentation
"""

__author__ = "Julien Lefebvre, Hugo Chauvary"
__email__ = 'numerai_2021@protonmail.com'

from catboost import CatBoostClassifier as cat
import numpy as np
import joblib

from common.config.constant import DATA_FOLDER, MACHINE_TYPE
from common.module.model.model import Model
from common.module.splitter import Splitter
from common.util.read_param import read_param


class Destroyai(Model):
    def __init__(self, X, y, pre_trained=False):
        # trained model as class attribute
        self.model = self.fit(X, y, pre_trained)


    def fit(self, X, y, pre_trained):
        # if pre_trained load model from joblib file
        if pre_trained:
            model = joblib.load(f'{DATA_FOLDER}/model/{type(self).__name__.lower()}.joblib')
            return model

        # split training dataset
        (X_train, X_test, y_train, y_test) = Splitter.split(X, y)

        _, counts = np.unique(y_train, return_counts=True)
        weights = np.asarray(1 - counts/len(y_train))

        # define the CatBoostClassifier model
        params = read_param(type(self).__name__.lower())
        catt = cat(class_weights=weights, task_type=MACHINE_TYPE, **params)

        # fit model
        estimator = catt.fit(X_train,
                             y_train,
                             eval_set=(X_test, y_test),
                             use_best_model=True)

        return estimator


    def predict(self, X_pred):
        yhat = self.model.predict(X_pred.x, verbose=True)
        return X_pred.ids, yhat/4


    def score(self, X_valid, y_valid):
        score = self.model.score(X_valid, y_valid)

        return score
