"""
lgbm_regressor.py

Implements Lgbm Regressor
Refer to model.model.py for documentation
"""

__author__ = "Julien Lefebvre, Hugo Chauvary"
__email__ = 'numerai_2021@protonmail.com'

from lightgbm import LGBMRegressor
from lightgbm import log_evaluation
import joblib

from common.config.constant import DATA_FOLDER, MACHINE_TYPE
from common.module.model.model import Model
from common.module.splitter import Splitter
from common.util.read_param import read_param


class MegaClitai(Model):
    def __init__(self, X, y, pre_trained=False):
        # trained model as class attribute
        self.model = self.fit(X, y, pre_trained)


    def fit(self, X, y, pre_trained):
        if pre_trained:
            model = joblib.load(f'{DATA_FOLDER}/model/{type(self).__name__.lower()}.joblib')
            return model

        # fetch parameters/initialize model
        params = read_param(type(self).__name__.lower())
        model = LGBMRegressor(device=MACHINE_TYPE, **params)

        (X_train, X_test, y_train, y_test) = Splitter.split(X, y)

        # callbacks=log_evaluation()
        estimator = model.fit(X_train,
                              y_train,
                              callbacks=[log_evaluation(period=50)],
                              eval_set=(X_test, y_test))

        return estimator


    def predict(self, X_pred):
        yhat = self.model.predict(X_pred.x)
        return X_pred.ids, yhat/4


    def score(self, X_valid, y_valid):
        score = self.model.score(X_valid, y_valid)

        return score
