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
from common.config.constant import (
    SIGNAL_TRAIN_DATA,
    DATA_FOLDER,
    SECRET_KEY,
    PUBLIC_ID
)
from common.util.get_model_id import get_model_id

import numerapi
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from dateutil.relativedelta import relativedelta, FR

def main():
    """
    Main function for building numerai signal predictions
    """
    logger = Logger().logger
    logger.info(f"Begin script")
    napi = numerapi.SignalsAPI(secret_key=SECRET_KEY,
                               public_id=PUBLIC_ID)

    # read latest signal dataset
    # be it from numerai's repository or file
    # df_numerai = pd.read_csv(SIGNAL_TRAIN_DATA)
    df_numerai = pd.read_csv(f"{DATA_FOLDER}/signal/signals_train_val_bbg.csv")

    # quick data transformations
    # friday_date to %Y%m%d format
    # splitting ticker
    df_numerai['friday_date'] = pd.to_datetime(df_numerai['friday_date'], format='%Y%m%d')
    df_numerai['short_ticker'] = df_numerai['bloomberg_ticker'].str.split(' ').str[0]

    # dev
    logger.info(f"Numerai data latest friday: {df_numerai.friday_date.max()}")

    # building a ticker table
    df_tickers = df_numerai\
        .groupby(['bloomberg_ticker', 'short_ticker'])\
        .size()\
        .reset_index()

    # read latest training data
    df_signal = pd.read_csv(f"{DATA_FOLDER}/signal/updated_training.csv")

    # quick transformation on friday_date
    df_signal = df_signal.rename(columns={"date": "friday_date", "symbol": "short_ticker"})
    df_signal['friday_date'] = pd.to_datetime(df_signal['friday_date'], format='%Y-%m-%d')  

    # dev
    logger.info(f"Home data latest friday: {df_signal.friday_date.max()}")

    # merge our feature data with Numerai targets
    df = pd.merge(df_signal,
                  df_numerai,
                  on=['friday_date','short_ticker'])\
        .set_index('friday_date')

    # encode categorical attributes
    labelencoder = LabelEncoder()
    df['encoded_ticker'] = labelencoder.fit_transform(df['short_ticker'])

    # persist encoded ticker in ticker table
    df_tickers = pd.merge(df_tickers,
                          df.groupby(['short_ticker', 'encoded_ticker'])\
                            .size()\
                            .reset_index(),
                          on=['short_ticker'])

    # select feature names
    features = df_signal.filter(like='FEATURE_').columns.to_list()
    features.extend(['encoded_ticker'])

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
              categorical_feature=['encoded_ticker'])

    # choose data as of most recent
    # friday in Signal data + 1 week.
    last_friday = df_numerai.friday_date.max()\
        + relativedelta(weekday=FR(2))
    # last_friday = datetime.now() + relativedelta(weekday=FR(1))
    last_friday = last_friday.strftime('%Y-%m-%d')

    # filter W+1 data
    df_live = df_signal[df_signal['friday_date'] == last_friday]

    # dev
    logger.info(f"Target friday: {df_live.friday_date.max()}")
    # logger.info(f"Target friday: {last_friday}")

    # retrieve ticker metadata
    df_live = pd.merge(df_live,
                       df_tickers,
                       on=['short_ticker'])

    # use it to predict our targets
    logger.info(f"Building signal data")
    df_live['signal'] = model.predict(df_live[features])

    # build and save signal data to csv
    logger.info(f"Saving")
    signal_filepath = f"{DATA_FOLDER}/signal/signal.csv"
    df_live[['bloomberg_ticker', 'signal']].to_csv(signal_filepath,
                                                   index=False)

    # upload signal data
    logger.info(f"Uploading")
    # napi.upload_predictions(signal_filepath,
    #                         model_id=get_model_id('sternburg'))

    logger.info(f"Success!")


if __name__ == '__main__':
    main()
