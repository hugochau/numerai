"""
get_aws_creds.py

Defines get_aws_creds
"""

__author__ = "Julien Lefebvre, Hugo Chauvary"
__email__ = 'numerai_2021@protonmail.com'

import pandas as pd

from common.config.constant import DATA_FOLDER


def get_aws_creds():
    """
    Get AWS creds from external file

    return:
        - output: dictionnary with creds
    """
    creds = pd.read_csv(f'{DATA_FOLDER}/credential/aws.csv')

    # parse dataframe
    output = {}
    output['access_key_id']=creds.iloc[0,2]
    output['secret_access_key']=creds.iloc[0,3]

    return output
