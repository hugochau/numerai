"""
data.py

Implements Data
"""

import numpy as np
import pandas as pd
import json

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
    TARGET_COL,
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
        return self.df.index.values.astype('str')


    # x
    @property
    def x(self):
        """
        View of features, x, as a numpy float array"
        """
        return self.df.iloc[:, 2:-1].values

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
    def load_csv(type: str, test: str, single_precision: bool = True):
        """
        Load numerai dataset.

        It includes train data by default. To work with tournament data only,
        set `include_train` to False.

        Set `single_precision` to True in order to have data in float32 (saves memory).
        """
        if single_precision:
            # read first 100 rows to scan types
            # then replace all float64 types with float32
            df_test = pd.read_csv(f'{DATA_FOLDER}/numerai{test}/numerai_{type}_data.csv',
                                  nrows=100,
                                  header=0,
                                  index_col=0)

            float_cols = [c for c in df_test if df_test[c].dtype == "float64"]
            float32_cols = {c: np.float32 for c in float_cols}

            # df = pd.read_csv(f'{DATA_FOLDER}/numerai/numerai_{type}_data.csv',
            #                     header=0,
            #                     index_col=0,
            #                     engine='c',
            #                     nrows=2000,
            #                     dtype=float32_cols)

            df = pd.read_csv(f'{DATA_FOLDER}/numerai{test}/numerai_{type}_data.csv',
                                header=0,
                                index_col=0,
                                engine='c',
                                dtype=float32_cols)

            # if include_tournament:
            #     tourn = pd.read_csv(TOURNAMENT_FILE,
            #                         header=0,
            #                         index_col=0,
            #                         engine='c',
            #                         dtype=float32_cols)

            #     # merge train and tournament data to single dataframe
            #     df = pd.concat([train, tourn], axis=0)

            # else:
            #     df = train
        else:
            # regular parsing, float64 will be used
            df = pd.read_csv(f'{DATA_FOLDER}/numerai{test}/numerai_{type}_data.csv',
                                header=0,
                                index_col=0)

            # if include_tournament:
            #     tourn = pd.read_csv(TOURNAMENT_FILE,
            #                         header=0,
            #                         index_col=0)
            #     # merge train and tournament data to single dataframe
            #     df = pd.concat([train, tourn], axis=0)
            # else:
            #     df = train

        # rename columns
        rename_map = {'data_type': 'region'}
        # intelligence
        for i in range(1, N_FEATURES_INTEL + 1):
            rename_map['feature_intelligence' + str(i)] = 'i' + str(i)
        # charisma
        for i in range(1, N_FEATURES_CHARI + 1):
            rename_map['feature_charisma' + str(i)] = 'c' + str(i)
        # strength
        for i in range(1, N_FEATURES_STREN + 1):
            rename_map['feature_strength' + str(i)] = 's' + str(i)
        # dexterity
        for i in range(1, N_FEATURES_DEXT + 1):
            rename_map['feature_dexterity' + str(i)] = 'd' + str(i)
        # constitution
        for i in range(1, N_FEATURES_CONST + 1):
            rename_map['feature_constitution' + str(i)] = 'p' + str(i)
        # wisdom
        for i in range(1, N_FEATURES_WISDO + 1):
            rename_map['feature_wisdom' + str(i)] = 'w' + str(i)

        df.rename(columns=rename_map, inplace=True)

        # convert era, region, and labels to np.float32 or np.float64 depending on the mode
        df['era'] = df['era'].map(ERA_STR_TO_FLOAT)
        df['region'] = df['region'].map(REGION_STR_TO_FLOAT)

        if single_precision:
            df.iloc[:, 0:2] = df.iloc[:, 0:2].astype('float32')
        else:
            df.iloc[:, 0:2] = df.iloc[:, 0:2].astype('float64')

        # # make sure memory is contiguous so that, e.g., data.x is a view
        df = df.copy()

        # to avoid copies we need the dtype of each column to be the same
        if df.dtypes.unique().size != 1:
            raise TypeError("dtype of each column should be the same")

        return Data(df)


    @staticmethod
    @log_item
    def load_parquet(type: str, test: str):
        """
        Load data object from parquet file; return Data

        args:
            - type: training/validation/tournament
            - test: load test dataset

        returns
            - Data object
        """
        # read the feature metadata and get the "small" feature set
        with open(f'{DATA_FOLDER}/numerai/features.json', "r") as f:
            feature_metadata = json.load(f)

        # select feature set
        features = feature_metadata["feature_sets"][FEATURE_SIZE]

        # read in just those features along with era and target columns
        read_columns = features + [TARGET_COL]

        df = pd.read_parquet(f'{DATA_FOLDER}/numerai{test}/numerai_{type}_data_int8.parquet',
                             columns=read_columns)

        # pare down the number of eras to every 4th era
        # brings weekly data back at monthly level
        # only for training dataset
        if type == 'training':
            every_4th_era = df[ERA_COL].unique()[::4]
            df = df[df[ERA_COL].isin(every_4th_era)]

        # columns = [c for c in df.columns if
        #     c.startswith("feature_")
        #     or c == 'target_nomi_20'
        # ]

        # df.drop([ERA_COL], axis=1)

        return Data(df)


    @staticmethod
    @log_item
    def load_hdf():
        """
        Load data object from hdf archive; return Data
        """
        df = pd.read_hdf(f'{DATA_FOLDER}/numerai_dataset.hdf',
                         key='numerox_data')

        return Data(df)
