"""
data.py

Implements Data
"""

import numpy as np
import pandas as pd
import json
from common import module

from common.config.constant import (
    DATA_FOLDER,
    # TOURNAMENT_FILE,
    TOURNAMENT_REGIONS,
    # TRAIN_FILE,
    N_FEATURES_INTEL,
    N_FEATURES_CHARI,
    N_FEATURES_STREN,
    N_FEATURES_DEXT,
    N_FEATURES_CONST,
    N_FEATURES_WISDO,
    REGION_STR_TO_FLOAT,
    REGION_INT_TO_STR,
    REGION_STR_TO_INT,
    ERA_STR_TO_FLOAT,
    ERA_INT_TO_STR,
    ERA_STR_TO_INT,
    ERA_COL,
    DATA_TYPE_COL,
    FEATURE_SIZE
)
from common.util.log_item import log_item


class Data:
    def __init__(self, df):
        self.df = df


    # id
    @property
    def ids(self):
        "Copy of ids as a numpy str array"
        # return self.df.index.values.astype('str')
        return self.df.iloc[:, 0].values


    # x
    @property
    def x(self):
        """
        View of features, x, as a numpy float array"
        """
        return self.df.iloc[:, 3:-1].values

    # y
    @property
    def y(self):
        """
        View of features, x, as a numpy float array"
        """
        y = self.df.iloc[:, -1:].values*4
        return y.astype(int)


    # era
    @property
    def era(self):
        """
        Copy of era as a 1d numpy str array
        """
        series = self.df['era'].map(ERA_INT_TO_STR)
        return series.values.astype(str)


    def eras_str2int(self, eras):
        """
        List with eras names (str) converted to int
        """
        e = []
        for era in eras:
            if era in ERA_STR_TO_INT:
                e.append(ERA_STR_TO_INT[era])
            else:
                e.append(era)
        return e


    def era_isin(self, eras):
        """
        Copy of data containing only eras in the iterable `eras`
        """
        eras = self.eras_str2int(eras)
        idx = self.df.era.isin(eras)
        # return self.df[self.df['era'].isin(eras)]
        # return self.df[idx]
        return self[idx]


    # region
    @property
    def region(self):
        """
        Copy of region as a 1d numpy str array
        """
        series = self.df['region'].map(REGION_INT_TO_STR)
        return series.values.astype(str)


    def regions_str2int(self, regions):
        """
        List with regions names (str) converted to int
        """
        r = []
        for region in regions:
            if region in REGION_STR_TO_INT:
                r.append(REGION_STR_TO_INT[region])
            else:
                r.append(region)
        return r


    def region_isin(self, regions):
        """
        Copy of data containing only regions in the iterable `regions`
        """
        regions = self.regions_str2int(regions)
        idx = self.df.region.isin(regions)
        # return self.df[idx]
        return self[idx]


    def __getitem__(self, index):
        """
        Data indexing

        """
        typidx = type(index)
        if isinstance(index, str):
            if index.startswith('era'):
                if len(index) < 4:
                    raise IndexError('length of era string index too short')
                return self.era_isin([index])
            else:
                if index in ('train', 'validation', 'test', 'live'):
                    return self.region_isin([index])
                elif index == 'tournament':
                    return self.region_isin(TOURNAMENT_REGIONS)
                else:
                    raise IndexError('string index not recognized')

        # elif isinstance(index, slice):

        #     # step check
        #     if index.step is not None:
        #         if not nx.isint(index.step):
        #             msg = "slice step size must be None or psotive integer"
        #             raise IndexError(msg)
        #         if index.step < 1:
        #             raise IndexError('slice step must be greater than 0')
        #         step = index.step
        #     else:
        #         step = 1

        #     ueras = self.unique_era().tolist()

        #     # start
        #     era1 = index.start
        #     idx1 = None
        #     if era1 is None:
        #         idx1 = 0
        #     elif not nx.isstring(era1) or not era1.startswith('era'):
        #         raise IndexError("slice elements must be strings like 'era23'")
        #     if idx1 is None:
        #         idx1 = ueras.index(era1)

        #     # end
        #     era2 = index.stop
        #     idx2 = None
        #     if era2 is None:
        #         idx2 = len(ueras) - 1
        #     elif not nx.isstring(era2) or not era2.startswith('era'):
        #         raise IndexError("slice elements must be strings like 'era23'")
        #     if idx2 is None:
        #         idx2 = ueras.index(era2)

        #     if idx1 > idx2:
        #         raise IndexError("slice cannot go from large to small era")

        #     # find eras in slice
        #     eras = []
        #     for ix in range(idx1, idx2 + 1, step):
        #         eras.append(ueras[ix])

        #     data = self.era_isin(eras)

        #     return data

        elif typidx is pd.Series or typidx is np.ndarray:

            return Data(self.df[index])

        else:

            raise IndexError('indexing type not recognized')


    def save_hdf(self, compress=False):
        """
        Save data as an hdf archive
        """
        if compress:
            self.df.to_hdf(f'{DATA_FOLDER}/numerai_dataset.hdf',
                           'numerox_data',
                           complib='zlib',
                           complevel=4)
        else:
            self.df.to_hdf(f'{DATA_FOLDER}/numerai_dataset.hdf',
                           key='numerox_data')


    @staticmethod
    @log_item
    def load_csv(type: str, test: str, single_precision: bool = False):
        """
        Load numerai v2 training dataset.

        Set `single_precision` to True in order to have data in float32 (saves memory).

        args:
            - type:
            - test: whether to load test data or not
            - single_precision:
        """
        if single_precision:
            # read first 100 rows to scan types
            # then replace all float64 types with float32
            df_test = pd.read_csv(f'{DATA_FOLDER}/numerai{test}/v2/numerai_{type}_data.csv',
                                  nrows=100,
                                  header=0,
                                  index_col=0)

            float_cols = [c for c in df_test if df_test[c].dtype == "float64"]
            float32_cols = {c: np.float32 for c in float_cols}

            df = pd.read_csv(f'{DATA_FOLDER}/numerai{test}/v2/numerai_{type}_data.csv',
                             header=0,
                             index_col=0,
                             engine='c',
                             dtype=float32_cols)

        else:
            # regular parsing, float64 will be used
            tp = pd.read_csv(f'{DATA_FOLDER}/numerai{test}/v2/numerai_{type}_data.csv',
                             iterator=True,
                             chunksize=10000,
                             low_memory=True,
                             engine='c',)

            df = pd.concat(tp, ignore_index=True)

        return Data(df)


    @staticmethod
    @log_item
    def load_parquet(version: str, type: str, test: str):
        """
        Load data object from parquet file; return Data

        args:
            - version: dataset version
            - type: training/validation/tournament
            - test: load test dataset

        returns
            - Data object
        """
        if version == 'v2':
            filepath = f'{DATA_FOLDER}/numerai{test}/{version}/numerai_{type}_data.parquet'
            df = pd.read_parquet(filepath)

            features = df.filter(like='feature_')\
                .columns\
                .to_list()

            read_columns = ['id', ERA_COL, DATA_TYPE_COL] + features +  ['target']

            df = df[read_columns]

            return Data(df)

        elif version == 'v3':
            # read the feature metadata and get the FEATURE_SIZE feature set
            with open(f'{DATA_FOLDER}/numerai/{version}/features.json', "r") as f:
                feature_metadata = json.load(f)

            # select feature set
            features = feature_metadata["feature_sets"][FEATURE_SIZE]

            # read in just those features along with era and target columns
            read_columns = ['id', ERA_COL, DATA_TYPE_COL] + features + ['target_nomi_20']

            filepath = f'{DATA_FOLDER}/numerai{test}/{version}/numerai_{type}_data_int8.parquet'
            df = pd.read_parquet(filepath,
                                 columns=read_columns)

            # pare down the number of eras to every 4th era
            # brings weekly data back at monthly level
            # only for training dataset
            # if type == 'training':
            #     every_4th_era = df[ERA_COL].unique()[::4]
            #     df = df[df[ERA_COL].isin(every_4th_era)]

            return Data(df.reset_index())

        else:
            # read the feature metadata and get the FEATURE_SIZE feature set
            with open(f'{DATA_FOLDER}/numerai/{version}/features.json', "r") as f:
                feature_metadata = json.load(f)

            # select feature set
            features = feature_metadata["feature_sets"][FEATURE_SIZE]

            # read in just those features along with era and target columns
            read_columns = ['id', ERA_COL, DATA_TYPE_COL] + features + ['target_nomi_v4_20']

            filepath = f'{DATA_FOLDER}/numerai{test}/{version}/{type}_int8.parquet'
            df = pd.read_parquet(filepath,
                                 columns=read_columns)

            # pare down the number of eras to every 4th era
            # brings weekly data back at monthly level
            # only for training dataset
            # if type == 'train':
            #     every_4th_era = df[ERA_COL].unique()[::4]
            #     df = df[df[ERA_COL].isin(every_4th_era)]

            return Data(df.reset_index())


    @staticmethod
    @log_item
    def load_hdf():
        """
        Load data object from hdf archive; return Data
        """
        df = pd.read_hdf(f'{DATA_FOLDER}/numerai_dataset.hdf',
                         key='numerox_data')

        return Data(df)
