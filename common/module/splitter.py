"""
splitter.py

Implements Splitter
"""

__author__ = "Julien Lefebvre, Hugo Chauvary"
__email__ = 'numerai_2021@protonmail.com'


from sklearn.model_selection import train_test_split
import numpy as np

class Splitter:
    def split(X: np.array, y:np.array) -> list:
        """
        Split arrays or matrices into random train and test subsets

        args:
            - X: training features
            - y: traning targets

        returns:
            - List containing train-test split of inputs
        """
        return train_test_split(X,
                                y,
                                test_size = .3,
                                random_state = 0)
