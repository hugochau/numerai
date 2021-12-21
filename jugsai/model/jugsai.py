"""
VotingClass.py

Ensemble methods for more confident predictions
using Catboost, Lightgbm, LogisticRegression
"""

__author__ = "Julien Lefebvre, Hugo Chauvary"
__email__ = 'numerai_2021@protonmail.com'


import joblib

from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier

from common.config.constant import DATA_FOLDER
# from module.cross_validator import CrossValidator
from common.util.read_param import read_param


class Jugsai:
    def __init__(self, X, y, pre_trained=False, filename=None):
        # trained model as class attribute
        self.model = self.fit(X, y, pre_trained, filename)


    def fit(self, X, y, pre_trained, filename):
        # if pre_trained load model from joblib file
        if pre_trained:
            model = joblib.load(f'{DATA_FOLDER}/model/{filename}')
            return model

        params_a = read_param('jugsai')
        predictor_a = LGBMClassifier(**params_a)

        # params_b = read_param('CatboostClass')
        # predictor_b = CatBoostClassifier(**params_b)
        predictor_b = CatBoostClassifier(eval_metric='MultiClass',
                                        iterations=3500,
                                        task_type="CPU",
                                        devices='0:1',
                                        depth=7,
                                        learning_rate=.01,
                                        # class_weights=weights,
                                        thread_count=-1)

        params_c = read_param('Logistic')
        predictor_c = LogisticRegression(**params_c)

        model = VotingClassifier(estimators = [
                                    ("lightgbm_class", predictor_a),
                                    ("catclass", predictor_b),
                                    ('logit', predictor_c)
                                ],
                                voting= 'soft',
                                n_jobs=-1
        )

        model.fit(X,y)

        return model


    def predict(self, X_pred):
        yhat = self.model.predict(X_pred.x)
        return X_pred.ids, yhat/4


    def score(self, X_valid, y_valid):
        score = self.model.score(X_valid, y_valid)

        return score
