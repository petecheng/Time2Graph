# -*- coding: utf-8 -*-
import numpy as np
import pickle
import torch.nn as nn
import torch.optim as optim
from .rnn.deep_models import LSTMClassifier, GRUClassifier
from .rnn.deep_utils import DeepDataloader, DeepDataset, train_RNNs
from .shapelet_utils import shapelet_distance
from .time_aware_shapelets import learn_time_aware_shapelets
from ..utils.base_utils import Debugger


class Time2GraphSequence(object):
    """
        Time2Sequence Model:
        that is, using shapelet sequence as the input of a Sequence Model
    """
    def __init__(self, K=100, C=1000, seg_length=30, warp=2, tflag=True,
                 hidden_size=64, output_size=64, dropout=0.1, gpu_enable=True,
                 model='lstm', batch_size=100, **kwargs):
        super(Time2GraphSequence, self).__init__()
        self.K = K
        self.C = C
        self.seg_length = seg_length
        self.warp = warp
        self.tflag = tflag
        self.model = model
        self.batch_size = batch_size
        self.gpu_enable = gpu_enable
        self.shapelets = None
        self.rnns = None
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout
        self.lr = kwargs.pop('lr', 1e-2)
        self.p = kwargs.pop('p', 2)
        self.alpha = kwargs.pop('alpha', 10.0)
        self.beta = kwargs.pop('beta', 5.0)
        self.debug = kwargs.pop('debug', True)
        self.measurement = kwargs.pop('measurement', 'gdtw')
        self.niter = kwargs.pop('niter', 10)
        self.n_sequences = kwargs.pop('n_sequences', 1)
        self.kwargs = kwargs
        assert self.n_sequences == 1
        Debugger.info_print('initialize t2g model with {}'.format(self.__dict__))

    def learn_shapelets(self, x, y, num_segment, data_size, num_batch):
        assert x.shape[1] == num_segment * self.seg_length
        if self.tflag:
            self.shapelets = learn_time_aware_shapelets(
                time_series_set=x, label=y, K=self.K, C=self.C, p=self.p,
                num_segment=num_segment, seg_length=self.seg_length, data_size=data_size,
                lr=self.lr, alpha=self.alpha, beta=self.beta, num_batch=num_batch,
                measurement=self.measurement, gpu_enable=self.gpu_enable, **self.kwargs)
        else:
            raise NotImplementedError()

    def retrieve_sequence(self, x, init):
        assert self.shapelets is not None
        if len(x.shape) == 2:
            x = x.reshape(x.shape[0], x.shape[1], 1)
        data_length = x.shape[1]
        shapelet_dist = shapelet_distance(
            time_series_set=x, shapelets=self.shapelets, seg_length=self.seg_length, tflag=self.tflag,
            tanh=self.kwargs.get('tanh', False), debug=self.debug, init=init, warp=self.warp,
            measurement=self.measurement)
        ret = []
        for k in range(shapelet_dist.shape[0]):
            sdist, sequences = shapelet_dist[k], []
            for i in range(self.n_sequences):
                tmp = []
                for j in range(sdist.shape[0]):
                    min_s = np.argsort(sdist[j, :]).reshape(-1)[i]
                    tmp.append(self.shapelets[min_s][0])
                sequences.append(np.concatenate(tmp, axis=0))
            ret.append(np.array(sequences).reshape(self.n_sequences, data_length, -1))
        return np.array(ret)

    def fit(self, X, Y, init):
        sequences = self.retrieve_sequence(x=X, init=init).reshape(X.shape[0], X.shape[1], -1)
        if self.model == 'lstm':
            self.rnns = LSTMClassifier(data_size=X.shape[-1], hidden_size=self.hidden_size,
                                       output_size=self.output_size, dropout=self.dropout)
        elif self.model == 'gru':
            self.rnns = GRUClassifier(data_size=X.shape[-1], hidden_size=self.hidden_size,
                                      output_size=self.output_size, dropout=self.dropout)
        else:
            raise NotImplementedError()
        self.rnns.double()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.rnns.parameters(), lr=self.lr)
        if self.gpu_enable:
            self.rnns.cuda()
            criterion.cuda()
        train_dataset = DeepDataset(x=sequences, y=Y)
        train_dataloader = DeepDataloader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                          num_workers=2)
        for epoch in range(self.niter):
            train_RNNs(epoch=epoch, dataloader=train_dataloader, rnn=self.rnns, criterion=criterion,
                       optimizer=optimizer, debug=self.debug, gpu_enable=self.gpu_enable)

    def predict(self, X, init):
        assert self.shapelets is not None, 'shapelets has not been learnt yet...'
        assert self.rnns is not None, 'classifier has not been learnt yet...'
        return self.rnns(self.retrieve_sequence(x=X, init=init), len(X))

    def dump_shapelets(self, fpath):
        pickle.dump(self.shapelets, open(fpath, 'wb'))

    def load_shapelets(self, fpath):
        self.shapelets = pickle.load(open(fpath, 'rb'))

    def save_model(self, fpath, **kwargs):
        pickle.dump(self.__dict__, open(fpath, 'wb'))

    def load_model(self, fpath, **kwargs):
        paras = pickle.load(open(fpath, 'rb'))
        for key, val in paras.items():
            self.__dict__[key] = val
