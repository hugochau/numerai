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
from model.catboost_regressor import CatboostRegre


def main():
    """
    Main function for training numerai models
    """
    logger = Logger().logger
    logger.info(f"Begin script")
    napi = Api()

    # parse CLI arg
    # format: modelname_algoname
    # ex: benchmarkai_logistic
    args = Parser.parse()
    logger.info(f"Selected model: {args.model}")
    # model = args.model.split('_')[0]
    algo = args.model.split('_')[1]
    algo = algo[0].upper() + algo[1:]

    # download current training datasets
    # only when args.test is set to None
    # if not args.test:
    #     logger.info(f"Download {args.data} training datasets")

    #     if args.data == 'legacy':
    #         napi.download_dataset()
    #     else:
    #         napi.download_new_dataset('training')

    # load training data
    logger.info(f"Read {args.data} training data")
    if args.data == 'legacy':
        dtrain = Data.load_csv('training', args.test)
    else:
        dtrain = Data.load_parquet('training', args.test)

    dtrain.df.info(memory_usage="deep")
    logger.info(f"Loaded {dtrain.df.shape} training")

    # train model
    logger.info(f"Training model")
    # model = eval(algo)(dtrain.x, dtrain.y)
    model = CatboostRegre(dtrain.x, dtrain.y)

    # free up memory
    del dtrain

    # # # load validation data
    # # logger.info(f"Read validation data")
    # # if args.data == 'legacy':
    # #     dval = Data.load_csv('training', args.test)
    # # else:
    # #     dval = Data.load_parquet('validation', args.test)

    # # dval.df.info(memory_usage="deep")
    # # logger.info(f"Loaded {dval.df.shape} validation")

    # # # evaluate model
    # # score = model.score(dval.x, dval.y)
    # # logger.info(f"Model score: {score}")

    # # # free up memory
    # # del dval

    # save and upload model to s3
    logger.info(f"Save and upload model")
    save_model(model.model, args.model)
    # S3().upload_file(args.model)


if __name__ == '__main__':
    main()
