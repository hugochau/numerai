"""
adapter.py

Use this class to enrich the log format with additional keywords
"""

__author__ = "Julien Lefebvre, Hugo Chauvary"
__email__ = 'numerai_2021@protonmail.com'

import logging

class Adapter(logging.Filter):
    def __init__(self, scope):
        """
        Class instantiation

        param:
            scope: 'str'
        """
        self.scope = scope


    def filter(self, record):
        """
        Adds scope attribute to logger output line

        param:
            record: str, logger output
        """
        record.scope = self.scope.upper()
        return True
