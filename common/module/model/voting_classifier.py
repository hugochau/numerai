# """
# VotingClass.py

# Ensemble methods for more confident predictions
# using Catboost, Lightgbm, LogisticRegression
# """

# __author__ = "Julien Lefebvre, Hugo Chauvary"
# __email__ = 'numerai_2021@protonmail.com'


# import joblib

# from sklearn import preprocessing
# from sklearn.linear_model import LogisticRegression
# import catboost as cat
# import lightgbm as lgb
# from sklearn.ensemble import VotingClassifier
# # from sklearn.metrics import accuracy_score

# from config.constant import DATA_FOLDER
# # from module.cross_validator import CrossValidator
# from util.read_param import read_param


# class VotingClass:
#     def __init__(self, X, y, pre_trained=False, filename=None):
#         # trained model as class attribute
#         self.model = self.fit(X, y, pre_trained, filename)


#     def fit(self, X, y, pre_trained, filename):
#         # if pre_trained load model from joblib file
#         if pre_trained:
#             model = joblib.load(f'{DATA_FOLDER}/model/{filename}')
#             return model

#         params_a = read_param('LgbmClass')
#         predictor_a = lgb.LGBMClassifier(**params_a)

#         params_b = read_param('CatboostClass')
#         predictor_b = cat.CatBoostClassifier(**params_b)

#         params_c = read_param('Logistic')
#         predictor_c = LogisticRegression(**params_c)

#         model = VotingClassifier(estimators = [
#                                     ("lightgbm_class", predictor_a),
#                                     ("catclass", predictor_b),
#                                     ('logit', predictor_c)
#                                 ],
#                                 voting= 'soft',
#                                 n_jobs=-1
#         )

#         model.fit(X,y)

#         return model


#     def predict(self, X_pred):
#         yhat = self.model.predict(X_pred.x)
#         return X_pred.ids, yhat/4


#     def score(self, X_valid, y_valid):
#         score = self.model.score(X_valid, y_valid)

#         return score
