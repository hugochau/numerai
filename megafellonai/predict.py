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
from model.megafellonai import MegaFellonai


def main():
    """
    Main function for building numerai predictions
    """
    logger = Logger().logger
    logger.info(f"Begin script")
    napi = Api()

    # parse CLI arg
    args = Parser.parse()
    modelname = 'megafellonai'
    datatype = 'v3'

    # download current training datasets
    # only when args.test is set to None
    if not args.test:
        logger.info(f"Download tournament dataset")
        napi.download_dataset(datatype, 'tournament')

    # load data
    logger.info(f"Read tournament data")
    dtour = Data.load_parquet(datatype, 'live', args.test)
    dtour.df.info(memory_usage="deep")
    logger.info(f"Loaded {dtour.df.shape} tournament")

    # load model from s3
    load_model(modelname, args.test)
    model = MegaFellonai(None, None, True)

    # compute predictions
    logger.info(f"Compute predictions")
    ids, yhat = model.predict(dtour)

    # free up memory
    del dtour

    # save and upload predictions
    logger.info(f"Save predictions")
    Prediction(ids, yhat).save()

    logger.info(f"Upload predictions to {modelname}")
    napi.upload_predictions(modelname, datatype)


if __name__ == '__main__':
    main()
