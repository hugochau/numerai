"""
parser.py

Defines Parser
"""

__author__ = "Julien Lefebvre, Hugo Chauvary"
__email__ = 'numerai_2021@protonmail.com'

import argparse

from common.util.log_item import log_item


class Parser:

    @staticmethod
    @log_item
    def parse() -> dict:
        """
        Parse and validate CLI arguments

        returns:
            -dict: validated CLI arguments
        """
        # define parser
        description = 'CLI arguments parser for numerai'
        parser = argparse.ArgumentParser(description=description)

        # add arguments
        parser.add_argument('model',
                            help='the model to load/train')
        parser.add_argument('data',
                            help='the data type to load: legacy/new')
        parser.add_argument('--test',
                           required=False,
                           action='store_const',
                           const='_test',
                           default='')

        # parse arguments
        args = parser.parse_args()

        return args
