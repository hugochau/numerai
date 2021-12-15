"""
catboost_regressor.py

Implements CatBoost Regressor
Refer to model.model.py for documentation
"""

__author__ = "Julien Lefebvre, Hugo Chauvary"
__email__ = 'numerai_2021@protonmail.com'

from catboost import CatBoostRegressor as cat
import joblib

from common.config.constant import DATA_FOLDER
from common.module.model.model import Model
from common.module.splitter import Splitter
from common.util.read_param import read_param


class Felinai(Model):
    def __init__(self, X, y, pre_trained=False, filename=None):
        # trained model as class attribute
        self.model = self.fit(X, y, pre_trained, filename)


    def fit(self, X, y, pre_trained, filename):
        if pre_trained:
            model = joblib.load(f'{DATA_FOLDER}/model/{filename}')
            return model

        # define the CatBoostClassifier model
        params = read_param(type(self).__name__)
        catt = cat(**params)

        (X_train, X_test, y_train, y_test) = Splitter.split(X, y)

        estimator = catt.fit(X_train,
                            y_train,
                            eval_set=(X_test, y_test),
                            use_best_model=True,
                            plot=True)

        return estimator


    def predict(self, X_pred):
        yhat = self.model.predict(X_pred.x)
        return X_pred.ids, yhat/4


    def score(self, X_valid, y_valid):
        score = self.model.score(X_valid, y_valid)

        return score
