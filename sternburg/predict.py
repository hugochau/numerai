"""
predict.py

Main script for uploading predictions
"""

__author__ = "Julien Lefebvre, Hugo Chauvary"
__email__ = 'numerai_2021@protonmail.com'

import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from common.module.logger import Logger
from common.module.aws import S3
from common.module import Parser
from common.config.constant import (
    SIGNAL_DATA_S3,
    DATA_FOLDER,
    SECRET_KEY,
    PUBLIC_ID
)
from common.util.get_ticker_map import get_ticker_map
from common.util.get_model_id import get_model_id

import numerapi
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.preprocessing import LabelEncoder
from dateutil.relativedelta import relativedelta, FR

def main():
    """
    Main function for building numerai signal predictions
    """
    # define a logger
    logger = Logger().logger
    logger.info(f"Begin script")

    # parse CLI arg
    args = Parser.parse()

    # get ticker mapping
    df_ticker = get_ticker_map()

    logger.info(f"Ticker data {df_ticker.shape}")

    # read latest signal dataset
    if not args.test:
        df_numerai = pd.read_csv(f"{SIGNAL_DATA_S3}/signals_train_val_bbg.csv")
    else:
        filepath = f"{DATA_FOLDER}/signal/signals_train_val_bbg.csv"
        try:
            df_numerai = pd.read_csv(filepath)

        except FileNotFoundError:
            logger.error(f"Could not find {filepath}")
            sys.exit()

    # quick operations
    # friday_date to %Y%m%d format
    df_numerai['friday_date'] = pd.to_datetime(df_numerai['friday_date'],
                                               format='%Y%m%d')

    logger.info(f"Numerai data {df_numerai.shape}: {df_numerai.friday_date.max()}")

    # merge our feature data with Numerai targets
    df_numerai = pd.merge(df_numerai,
                          df_ticker,
                          on=['bloomberg_ticker'])

    logger.info(f"Merged Numerai/Ticker data {df_numerai.shape}: {df_numerai.friday_date.max()}")

    # read latest feature data
    if not args.test:
        sss = S3()
        sss.download_file(f'updated_training.csv', 'signalsdata', 'signal')

    df_feature = pd.read_csv(f"{DATA_FOLDER}/signal/updated_training.csv")

    # quick operations
    # friday_date to %Y%m%d format
    # columns renaming
    columns = {
        "date": "friday_date",
        "ticker": "yahoo_ticker"
    }
    df_feature = df_feature.rename(columns=columns)
    df_feature['friday_date'] = pd.to_datetime(df_feature['friday_date'],
                                              format='%Y-%m-%d')

    logger.info(f"Feature data {df_feature.shape}: {df_feature.friday_date.max()}")

    df_feature = pd.merge(df_feature,
                          df_ticker,
                          on=['yahoo_ticker'])

    logger.info(f"Merged Feature/Ticker data {df_feature.shape}")

    # encode categorical attributes
    labelencoder = LabelEncoder()
    df_feature['ticker_encoded'] = labelencoder.fit_transform(df_feature['yahoo_ticker'])

    # merge our feature data with Numerai targets
    df = pd.merge(df_feature,
                  df_numerai,
                  on=['friday_date','yahoo_ticker'])\
        .set_index('friday_date')

    logger.info(f"Merged Numerai/Feature data {df.shape}: {df_feature.friday_date.max()}")

    # drop na
    df = df.dropna(subset=['target_20d'])
    logger.info(f"Merged non null Numerai/Feature data {df.shape}: {df_feature.friday_date.max()}")

    # select features
    features = df_feature.filter(like='FEATURE_')\
        .columns\
        .to_list()
    features.append('ticker_encoded')

    # fit lightGBM model
    params = {
        "metric": "MAE",
        "lambda_l1": 5.478607699138969,
        "lambda_l2": 1.9463954966660213e-05,
        "num_leaves": 86,
        "feature_fraction": 0.6442004065985856,
        "bagging_fraction": 0.6732807915469317,
        "bagging_freq": 6,
        "min_child_samples": 22
    }

    model = LGBMRegressor(**params)
    model.fit(df[features],
              df['target_20d'],
              categorical_feature=['ticker_encoded'])

    # choose data as of most recent
    # friday in Signal data + 1 week.
    last_friday = df_numerai.friday_date.max()\
        + relativedelta(weekday=FR(2))
    # last_friday = df_numerai.friday_date.max()
    last_friday = last_friday.strftime('%Y-%m-%d')

    # filter W+1 data
    df_live = df_feature[df_feature['friday_date'] == last_friday]\
        .copy(deep=True)

    logger.info(f"Target friday: {df_live.friday_date.max()}")
    logger.info(f"Live data {df_live.shape}")

    # use it to predict our targets
    logger.info(f"Building signal data")
    df_live['signal'] = model.predict(df_live[features])

    logger.info(f"Live data {df_live.shape}")

    # build and save signal data to csv
    logger.info(f"Saving")
    signal_filepath = f"{DATA_FOLDER}/predictions.csv"
    df_live[['bloomberg_ticker', 'signal']].to_csv(signal_filepath,
                                                   index=False)

    # upload signal data
    if not args.test:
        model_name = 'sternburg'
        logger.info(f"Uploading {model_name}")
        napi = numerapi.SignalsAPI(secret_key=SECRET_KEY,
                                public_id=PUBLIC_ID)
        napi.upload_predictions(signal_filepath,
                                model_id=get_model_id(model_name))

    logger.info(f"Success!")


if __name__ == '__main__':
    main()
