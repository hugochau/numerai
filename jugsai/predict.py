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

from common.module import Parser, Api, Data, Prediction
from common.module.logger import Logger
from common.module.api import Api
from common.util.load_model import load_model
from model.jugsai import Jugsai


def main():
    """
    Main function for building numerai predictions
    """
    logger = Logger().logger
    logger.info(f"Begin script")
    napi = Api()

    # parse CLI arg
    # format: modelname
    # ex: benchmarkai
    args = Parser.parse()

    # download current training datasets
    # only when args.test is set to None
    # if not args.test:
    #     logger.info(f"Download {args.data} tournament dataset")

    #     if args.data == 'legacy':
    #         napi.download_dataset()
    #     else:
    #         napi.download_new_dataset('tournament')

    # load data
    logger.info(f"Read {args.data} tournament data")
    if args.data == 'legacy':
        dtour = Data.load_csv('tournament', args.test)
    else:
        dtour = Data.load_parquet('tournament', args.test)
    print(dtour.df.head())

    # dtour.df.info(memory_usage="deep")
    # logger.info(f"Loaded {dtour.df.shape} tournament")

    # load model from s3
    # load_model(args.model)
    # model = CatboostRegre(None, None, True, 'felinai_catboostRegre.joblib')

    # # compute predictions
    # logger.info(f"Compute predictions")
    # ids, yhat = model.predict(dtour)

    # # free up memory
    # del dtour

    # save and upload predictions
    # logger.info(f"Save predictions")
    # Prediction(ids, yhat).save()

    # logger.info(f"Upload predictions")
    # napi.upload_predictions(args.model, args.data)

    # upload predictions/diagnostic
    # if args.data == 'new':

    #     napi.upload_diagnostics(args.model)


if __name__ == '__main__':
    main()
