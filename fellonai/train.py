"""
main.py

Main script for training models
"""
__author__ = "Julien Lefebvre, Hugo Chauvary"
__email__ = 'numerai_2021@protonmail.com'

import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from common.module import Parser, Api, Data
from common.module.aws import S3
from common.module.logger import Logger
from common.util.save_model import save_model
from model.fellonai import Fellonai


def main():
    """
    Main function for training numerai models
    """
    logger = Logger().logger
    logger.info(f"Begin script")

    # parse CLI arg
    args = Parser.parse()
    modelname = 'fellonai'
    logger.info(f"Selected model:{modelname}")

    # download current training datasets
    # only when args.test is set to None
    if not args.test:
        logger.info(f"Download training datasets")
        Api().download_dataset()

    # load training data
    logger.info(f"Read training data")
    dtrain = Data.load_csv('training', args.test)
    dtrain.df.info(memory_usage="deep")
    logger.info(f"Loaded {dtrain.df.shape} training")

    # train model
    logger.info(f"Training model")
    model = Fellonai(dtrain.x, dtrain.y)

    # free up memory
    del dtrain

    # save and upload model to s3
    logger.info(f"Save and upload model")
    save_model(model.model, modelname)
    S3().upload_file(modelname)


if __name__ == '__main__':
    main()
