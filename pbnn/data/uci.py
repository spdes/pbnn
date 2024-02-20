"""
Implementation credits to Sebastian.
"""
import os
import pandas
import numpy as np
import abc
from pbnn.data import DataSet
from numpy.random import RandomState
from enum import Enum
from jax.nn import one_hot
from sklearn.datasets import load_svmlight_file


class UCIData(DataSet):
    type: str
    nlabel: int = 1

    def __init__(self, data_path: str, rng: RandomState):
        data = self.load_data(data_path)
        data = rng.permutation(data)

        xs = self.standardise(data[:, :-1])
        self.input_dim = xs.shape[1]
        if self.type == 'regression':
            ys = self.reshape(data[:, -1])
            ys = self.standardise(ys)
        else:
            ys = one_hot(data[:, -1], self.nlabel)

        data_size, _ = data.shape
        self.n = int(data_size * 0.6)
        self.n_val = int(data_size * 0.3)
        self.n_test = data_size - (self.n + self.n_val)

        # Training
        self.xs = xs[:self.n]
        self.ys = ys[:self.n]

        # Validation
        self.val_xs = xs[self.n:self.n + self.n_val]
        self.val_ys = ys[self.n:self.n + self.n_val]

        # Test
        self.test_xs = xs[-self.n_test:]
        self.test_ys = ys[-self.n_test:]

        self.xi = 1.

    @abc.abstractmethod
    def load_data(self, data_path: str):
        pass


class Boston(UCIData):
    type = 'regression'

    def load_data(self, data_path: str):
        return pandas.read_fwf(os.path.join(data_path, 'housing.data'), header=None).values


class Concrete(UCIData):
    type = 'regression'

    def load_data(self, data_path: str):
        return pandas.read_excel(os.path.join(data_path, 'Concrete_Data.xls')).values


class Energy(UCIData):
    type = 'regression'

    def load_data(self, data_path: str):
        data = pandas.read_excel(os.path.join(data_path, 'ENB2012_data.xlsx')).values
        # energy has two targets, we use the first one
        data = data[:, :-1]
        return data


class Kin8(UCIData):
    type = 'regression'

    def load_data(self, data_path: str):
        return pandas.read_csv(os.path.join(data_path, 'dataset_2175_kin8nm.csv')).values


class Naval(UCIData):
    type = 'regression'

    def load_data(self, data_path: str):
        data = pandas.read_fwf(os.path.join(data_path, 'UCI CBM Dataset', 'data.txt'), header=None).values
        # target is in the second last column, thus drop the last
        data = data[:, :-1]
        # dims 8 and 11 have std=0
        data = np.delete(data, [8, 11], axis=1)
        return data


class Yacht(UCIData):
    type = 'regression'

    def load_data(self, data_path: str):
        data = pandas.read_fwf(os.path.join(data_path, 'yacht_hydrodynamics.data'), header=None).values
        return data


class Protein(UCIData):
    type = 'regression'

    def load_data(self, data_path: str):
        return pandas.read_csv(os.path.join(data_path, 'CASP.csv')).values


class WineRed(UCIData):
    type = 'regression'

    def load_data(self, data_path: str):
        return pandas.read_csv(os.path.join(data_path, 'winequality-red.csv'), delimiter=';').values


class WineWhite(UCIData):
    type = 'regression'

    def load_data(self, data_path: str):
        return pandas.read_csv(os.path.join(data_path, 'winequality-white.csv'), delimiter=';').values


class Power(UCIData):
    type = 'regression'

    def load_data(self, data_path: str):
        return pandas.read_excel(os.path.join(data_path, 'CCPP', 'Folds5x2_pp.xlsx')).values


class YearPredictionMSD(UCIData):
    # note that this data set comes with a standard train/test split
    """
    You should respect the following train / test split:
    train: first 463,715 examples
    test: last 51,630 examples
    """
    type = 'regression'

    def load_data(self, data_path: str):
        return pandas.read_csv(os.path.join(data_path, 'YearPredictionMSD.txt'), header=None).values


class Australian(UCIData):
    type = 'classification'

    def load_data(self, data_path: str):
        self.nlabel = 2
        if '~' in data_path:
            data_path = os.path.expanduser(data_path)
        X, y = load_svmlight_file(os.path.join(data_path, 'australian'))
        X = np.array(X.todense())
        y = (y.reshape(-1, 1) + 1) / 2
        data = np.hstack((X, y))
        return data


class Cancer(UCIData):
    type = 'classification'

    def load_data(self, data_path: str):
        self.nlabel = 2
        data = pandas.read_csv(os.path.join(data_path, 'wdbc.data'), header=None)
        # benign -> 0
        # malignant -> 1
        y = data[1].replace('B', 0).replace('M', 1).values.reshape(-1, 1)
        # first column is ID
        # second column is target
        X = data.loc[:, 2:].values
        data = np.hstack((X, y))
        return data


class Ionosphere(UCIData):
    type = 'classification'

    def load_data(self, data_path: str):
        self.nlabel = 2
        data = pandas.read_csv(os.path.join(data_path, 'ionosphere.data'), header=None)
        # The second column is all zero, delete
        data = data.replace('g', 0).replace('b', 1).values
        return np.delete(data, 1, axis=1)


class Glass(UCIData):
    type = 'classification'

    def load_data(self, data_path: str):
        self.nlabel = 6
        data = pandas.read_csv(os.path.join(data_path, 'glass.data'), header=None).values
        # the first column is the ID
        X = data[:, 1:-1]
        y = data[:, -1].reshape(-1, 1)
        # the target is 1..7 but there is no 4 in the data set
        mask = y > 4
        y[mask] = y[mask] - 1
        # start with 0
        y = y - 1
        data = np.hstack((X, y))
        return data


class Vehicle(UCIData):
    type = 'classification'

    def load_data(self, data_path: str):
        raise NotImplementedError


class Waveform(UCIData):
    type = 'classification'

    def load_data(self, data_path: str):
        raise NotImplementedError


class Digits(UCIData):
    type = 'classification'

    def load_data(self, data_path: str):
        raise NotImplementedError


class Satellite(UCIData):
    type = 'classification'

    def load_data(self, data_path: str):
        self.nlabel = 6
        trn = pandas.read_csv(os.path.join(data_path, 'sat.trn'), delimiter=' ', header=None).values
        tst = pandas.read_csv(os.path.join(data_path, 'sat.tst'), delimiter=' ', header=None).values
        data = np.vstack((trn, tst)).astype(np.float64)
        X = data[:, :-1]
        y = data[:, -1].reshape(-1, 1)
        # the target is 1..7 but there is no 6 in the data set
        mask = y > 5
        y[mask] = y[mask] - 1
        # start with 0
        y = y - 1
        data = np.hstack((X, y))
        return data


class UCIEnum(Enum):
    boston = Boston
    concrete = Concrete
    energy = Energy
    kin8 = Kin8
    naval = Naval
    yacht = Yacht
    protein = Protein
    winered = WineRed
    winewhite = WineWhite
    power = Power

    australian = Australian
    cancer = Cancer
    ionosphere = Ionosphere
    glass = Glass
    satellite = Satellite


"""
# debug
from pbnn.data.uci import UCIEnum
from numpy.random import RandomState


for z in UCIEnum:
    d = z.value('/home/zgbkdlm/Research/fo/experiments/data', RandomState(666))
    print(type(d).__name__, f'x: {d.xs.shape}', f'y: {d.ys.shape}')

"""
