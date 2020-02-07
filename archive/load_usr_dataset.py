# -*- coding: utf-8 -*-
import pandas
from config import *


def load_usr_dataset_by_name(fname, length):
    """
    load UCR dataset given dataset name.
    :param fname:
        dataset name, e.g., Earthquakes.
    :param length:
        time series length that want to load in.
    :return:
    """
    dir_path = '{}/dataset/UCRArchive_2018'.format(module_path)
    assert path.isfile('{}/{}/{}_TEST.tsv'.format(dir_path, fname, fname)), '{} NOT EXIST in UCR!'.format(fname)
    train_data = pandas.read_csv('{}/{}/{}_TRAIN.tsv'.format(dir_path, fname, fname), sep='\t', header=None)
    test_data = pandas.read_csv('{}/{}/{}_TEST.tsv'.format(dir_path, fname, fname), sep='\t', header=None)
    init = train_data.shape[1] - length
    x_train, y_train = train_data.values[:, init:].astype(np.float).reshape(-1, length, 1), \
                       train_data[0].values.astype(np.int)
    x_test, y_test = test_data.values[:, init:].astype(np.float).reshape(-1, length, 1), \
                     test_data[0].values.astype(np.int)
    lbs = np.unique(y_train)
    y_train_return, y_test_return = np.copy(y_train), np.copy(y_test)
    for idx, val in enumerate(lbs):
        y_train_return[y_train == val] = idx
        y_test_return[y_test == val] = idx
    return x_train, y_train_return, x_test, y_test_return


