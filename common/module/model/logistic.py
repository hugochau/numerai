# """
# logistic.py

# Implements Logistic Regressor
# Refer to model.model.py for documentation
# """

# __author__ = "Julien Lefebvre, Hugo Chauvary"
# __email__ = 'numerai_2021@protonmail.com'


# from sklearn.linear_model import LogisticRegression
# import joblib

# from common.config.constant import DATA_FOLDER
# from common.module.model.model import Model
# from common.util.read_param import read_param


# class Logistic(Model):
#     def __init__(self, X, y, pre_trained: bool = False, filename = None) -> None:
#         self.model = self.fit(X, y, pre_trained, filename)


#     def fit(self, X, y, pre_trained, filename):
#         if pre_trained:
#             print(f'{DATA_FOLDER}/model/{filename}')
#             model = joblib.load(f'{DATA_FOLDER}/model/{filename}')
#             return model

#         # fetch parameters/initialize model
#         params = read_param(type(self).__name__)
#         model = LogisticRegression(**params)

#         model.fit(X, y)

#         return model


#     def score(self, X_valid, y_valid):
#         score = self.model.score(X_valid, y_valid)

#         return score


#     def predict(self, X_pred):
#         yhat = self.model.predict(X_pred.x)
#         return X_pred.ids, yhat/4
