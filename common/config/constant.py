"""
constant.py

Define globals here
"""
import logging
import datetime
import os

import pandas as pd
pd.set_option('display.max_colwidth', 1000)



# data
DATA_FOLDER = 'data'

# logger
LOG_LEVEL = logging.INFO
LOG_FOLDER = os.path.join(DATA_FOLDER, 'log')
LOG_FILENAME = f'log_{datetime.datetime.isoformat(datetime.datetime.today())}'
LOG_FILEPATH = os.path.join(LOG_FOLDER, LOG_FILENAME)

# features
N_FEATURES_INTEL = 12
N_FEATURES_CHARI = 86
N_FEATURES_STREN = 38
N_FEATURES_DEXT = 14
N_FEATURES_CONST = 114
N_FEATURES_WISDO = 46
N_FEATURES = 310

REGION_STR_TO_FLOAT = {'train': 0., 'validation': 1., 'test': 2., 'live': 3.}
REGION_STR_TO_INT = {'train': 0, 'validation': 1, 'test': 2, 'live': 3}
REGION_INT_TO_STR = {0: 'train', 1: 'validation', 2: 'test', 3: 'live'}

ERA_INT_TO_STR = {}
ERA_STR_TO_INT = {}
ERA_STR_TO_FLOAT = {}
for i in range(998):
    name = 'era' + str(i)
    ERA_INT_TO_STR[i] = name
    ERA_STR_TO_INT[name] = i
    ERA_STR_TO_FLOAT[name] = float(i)
ERA_INT_TO_STR[999] = 'eraX'
ERA_STR_TO_INT['eraX'] = 999
ERA_STR_TO_FLOAT['eraX'] = 999.0

TOURNAMENT_REGIONS = ['validation', 'test', 'live']

def read_cred(data_folder):
    """
    """
    output = {}
    df = pd.read_csv(f'{data_folder}/credential/numerai.csv')

    output['public_id'] = df['public_id'].to_string(index=False).strip()
    output['secret_key'] = df['secret_key'].to_string(index=False).strip()

    return output

CREDS = read_cred(DATA_FOLDER)

# api keys
SECRET_KEY = CREDS['secret_key']
PUBLIC_ID = CREDS['public_id']

DFILES = {
    'tournament': {
        'numerai_tournament_data_int8.parquet',
        'features.json'
    },
    'training': {
        'numerai_training_data_int8.parquet',
        'numerai_validation_data_int8.parquet',
        'features.json'
    }
}


AWS_REGION = 'us-east-1'

ERA_COL = "era"
TARGET_COL = "target_nomi_20"
DATA_TYPE_COL = "data_type"
FEATURE_SIZE = 'medium'
