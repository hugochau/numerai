# """
# xgb_classifier.py

# XGB Classifier model
# """

# from os import read
# from xgboost import XGBClassifier as xgb
# from sklearn import preprocessing
# from sklearn.metrics import accuracy_score
# import joblib

# from config.constant import DATA_FOLDER

# from module.cross_validator import CrossValidator

# from util.read_param import read_param


# class XgbClass:
#     def __init__(self, X, y, pre_trained=False, filename=None):
#         # trained model as class attribute
#         self.model = self.fit(X, y, pre_trained, filename)


#     def fit(self, X, y, pre_trained, filename):
#         if pre_trained:
#             model = joblib.load(f'{DATA_FOLDER}/model/{filename}')
#             return model

#         model = xgb(early_stopping_rounds=5,
#                     use_label_encoder=False)

#         # fetch parameters/initialize model
#         params = read_param(type(self).__name__)
#         estimator = CrossValidator.grid_search(model, params)

#         # train model
#         estimator.fit(X, y)

#         return estimator


#     # @log_item
#     def predict(self, X_pred):
#         """
#         """
#         yhat = self.model.predict(X_pred.x)
#         return X_pred.ids, yhat/4


#     def score(self, X_valid, y_valid):
#         # make predictions for validation data
#         y_pred = self.model.predict(X_valid)
#         predictions = [round(value) for value in y_pred]

#         # evaluate predictions
#         score = accuracy_score(y_valid, predictions)

#         return score
