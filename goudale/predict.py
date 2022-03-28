"""
predict.py

Main script for uploading predictions
"""

__author__ = "Julien Lefebvre, Hugo Chauvary"
__email__ = 'numerai_2021@protonmail.com'

from operator import index
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from common.module import SignalApi, Splitter # Api
from common.module.logger import Logger
from common.config.constant import SIGNAL_TRAIN_DATA, MACHINE_TYPE

import pandas as pd
from catboost import CatBoostRegressor as cat
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from dateutil.relativedelta import relativedelta, FR

def main():
    """
    Main function for building numerai signal predictions
    """
    logger = Logger().logger
    logger.info(f"Begin script")
    napi = SignalApi()

    # read latest signal dataset
    # be it from numerai's repository or file
    # df_numerai = pd.read_csv(SIGNAL_TRAIN_DATA)
    df_numerai = pd.read_csv(f'sternburg/signals_train_val_bbg.csv')

    # quick operations
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

    # # columns = ['friday_date', 'short_ticker', 'data_type', 'target_20d']
    # # df_numerai = df_numerai[columns]

    # read latest training data
    df_signal = pd.read_csv(f'sternburg/updated_training.csv')

    # quick transformation on friday_date
    df_signal = df_signal.rename(columns={"date": "friday_date", "symbol": "short_ticker"})
    df_signal['friday_date'] = pd.to_datetime(df_signal['friday_date'], format='%Y-%m-%d')  

    logger.info(f"Home data latest friday: {df_signal.friday_date.max()}")

    # merge our feature data with Numerai targets
    df = pd.merge(df_signal,
                  df_numerai,
                  on=['friday_date','short_ticker'])\
        .set_index('friday_date')

    # encode categorical attributes
    # labelencoder = LabelEncoder()
    # df['encoded_ticker'] = labelencoder.fit_transform(df['short_ticker'])

    # # persist encoded ticker in ticker table
    # df_tickers = pd.merge(df_tickers,
    #                       df.groupby(['short_ticker', 'encoded_ticker'])\
    #                         .size()\
    #                         .reset_index(),
    #                       on=['short_ticker'])

    # select feature names
    features = df_signal.filter(like='FEATURE_').columns.to_list()
    # features.extend(['encoded_ticker'])

    # fit lightGBM model
    params = {
        "thread_count": -1,
        "iterations": 3500,
        "learning_rate": 0.009,
        "max_depth": 7,
        "num_leaves": 128
    }

    model = cat(task_type=MACHINE_TYPE,
                **params)

    (X_train, X_test, y_train, y_test) = Splitter.split(df[features],
                                                        df['target_20d'])

    model = model.fit(X_train,
                      y_train,
                      eval_set=(X_test, y_test),
                      use_best_model=True,
                      plot=True)
    
    # model.fit(df[features],
    #           df['target_20d'],
    #           categorical_feature=['encoded_ticker'])

    # choose data as of most recent
    # friday in Signal data + 1 week.
    last_friday = datetime.now() + relativedelta(weekday=FR(-2))
    date_string = last_friday.strftime('%Y-%m-%d')
    # this would not work as df_signal
    # might include newer data already
    # friday = df_signal.friday_date.max()

    # filter W+1 data
    df_live = df_signal[df_signal['friday_date'] == date_string]

    # retrieve ticker metadata
    df_live = pd.merge(df_live,
                       df_tickers,
                       on=['short_ticker'])

    # use it to predict our targets
    logger.info(f"Building signal data")
    df_live['signal'] = model.predict(df_live[features])

    # build and save signal data to csv
    df_live = df_live[['bloomberg_ticker', 'signal']]
    df_live.to_csv('signal.csv', index=False)

    # upload signal data
    logger.info(f"Uploading")
    # napi.upload_predictions('sternburg')

    logger.info(f"Success!")


if __name__ == '__main__':
    main()
