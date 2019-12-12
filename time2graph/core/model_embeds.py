# -*- coding: utf-8 -*-
import numpy as np
import pickle
from copy import deepcopy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize
from .time_aware_shapelets import learn_time_aware_shapelets
from .shapelet_embedding import ShapeletEmbedding
from ..utils.base_utils import ModelUtils, Debugger


class Time2GraphEmbed(ModelUtils):
    """
        Time2Graph model
        Hyper-parameters:
            K: number of learned shapelets
            C: number of candidates
            A: number of shapelets assigned to each segment
            tflag: timing flag
            opt_metric: optimal metric using in outside-classifier
    """
    def __init__(self, kernel, K=100, C=1000, seg_length=30, warp=2, tflag=True,
                 gpu_enable=True, percentile=15, opt_metric='f1', mode='aggregate',
                 batch_size=100, **kwargs):
        super(Time2GraphEmbed, self).__init__(kernel=kernel, **kwargs)
        self.K = K
        self.C = C
        self.seg_length = seg_length
        self.warp = warp
        self.tflag = tflag
        self.opt_metric = opt_metric
        self.mode = mode
        self.batch_size = batch_size
        self.gpu_enable = gpu_enable
        self.percentile = percentile
        self.shapelets = None
        self.sembeds = None
        self.clf = None
        self.lr = kwargs.pop('lr', 1e-2)
        self.p = kwargs.pop('p', 2)
        self.alpha = kwargs.pop('alpha', 0.1)
        self.beta = kwargs.pop('beta', 0.05)
        self.multi_graph = kwargs.pop('multi_graph', False)
        self.debug = kwargs.pop('debug', True)
        self.measurement = kwargs.pop('measurement', 'gdtw')
        self.verbose = kwargs.pop('verbose', False)
        self.kwargs = kwargs
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

    def fit_embedding_model(self, x, y, cache_dir, init=0):
        assert self.shapelets is not None, 'shapelets has not been learnt yet'
        self.sembeds = ShapeletEmbedding(
            seg_length=self.seg_length, tflag=self.tflag, multi_graph=self.multi_graph,
            cache_dir=cache_dir, tanh=self.kwargs.get('tanh', False), debug=self.debug,
            percentile=self.percentile, measurement=self.measurement, mode=self.mode,
            **self.kwargs)
        self.sembeds.fit(time_series_set=x[np.argwhere(y == 0).reshape(-1), :, :],
                         shapelets=self.shapelets, warp=self.warp, init=init)

    def embed(self, x, init=0):
        assert self.sembeds is not None, 'shapelet-embedding model has not been learnt yet'
        return self.sembeds.time_series_embedding(time_series_set=x, shapelets=self.shapelets, warp=self.warp, init=init)

    def set_deepwalk_args(self, **dw_args):
        for key, val in dw_args.items():
            self.kwargs[key] = val

    def fit(self, X, Y, n_splits=5, init=0, reset=True, balanced=True, norm=False,
            cache_dir='./', **kwargs):
        num_segment = int(X.shape[1] / self.seg_length)
        data_size = X.shape[-1]
        if reset or self.shapelets is None:
            self.learn_shapelets(
                x=X, y=Y, num_segment=num_segment, data_size=data_size, num_batch=X.shape[0] // self.batch_size)
        if reset or self.sembeds is None:
            Debugger.info_print('fit embedding model...')
            self.fit_embedding_model(x=X, y=Y, cache_dir=cache_dir, init=init)
        max_clf_args, max_metric, clf = None, -1, self.clf__()
        embeds = self.sembeds.time_series_embedding(
            time_series_set=X, shapelets=self.shapelets,
            warp=self.warp, init=init)
        if norm:
            embeds = normalize(embeds, axis=0)
        Debugger.info_print('{} paras to be tuned'.format(self.para_len(balanced=balanced)))
        arguments = self.clf_paras(balanced=balanced)
        arg_size, cnt = self.para_len(balanced=balanced), 0.0
        metric_method = self.return_metric_method(opt_metric=self.opt_metric)

        tuning, opt_args = kwargs.get('tuning', True), kwargs.get('opt_args', None)
        if tuning:
            Debugger.info_print('running parameter tuning for fit...')
            max_accu, max_prec, max_recall, max_f1, max_clf_model = -1, -1, -1, -1, None
            for args in arguments:
                clf.set_params(**args)
                Debugger.debug_print(msg='{:.2f}% inner args tuned; args: {}'.format(cnt * 100.0 / arg_size, args),
                                     debug=self.debug)
                skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
                tmp, accu, prec, recall, f1 = 0, 0, 0, 0, 0
                for train_idx, test_idx in skf.split(embeds, Y):
                    clf.fit(embeds[train_idx], Y[train_idx])
                    y_true, y_pred = Y[test_idx], clf.predict(embeds[test_idx])
                    tmp += metric_method(y_true=y_true, y_pred=y_pred)
                    accu += accuracy_score(y_true=y_true, y_pred=y_pred)
                    prec += precision_score(y_true=y_true, y_pred=y_pred)
                    recall += recall_score(y_true=y_true, y_pred=y_pred)
                    f1 += f1_score(y_true=y_true, y_pred=y_pred)
                tmp /= n_splits
                accu /= n_splits
                prec /= n_splits
                recall /= n_splits
                f1 /= n_splits
                if max_metric < tmp:
                    max_metric, max_clf_args, max_clf_model = tmp, args, deepcopy(clf)
                    max_accu, max_prec, max_recall, max_f1 = accu, prec, recall, f1
                cnt += 1.0
            if self.verbose:
                Debugger.info_print('args {} for clf {}-{}, performance: {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(
                    max_clf_args, self.kernel, self.opt_metric, max_accu, max_prec, max_recall, max_f1))
            self.clf = {'clf': max_clf_model, 'clf-args': max_clf_args}
        else:
            assert opt_args is not None, 'missing opt args specified'
            clf.set_params(**opt_args)
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
            tmp = np.zeros(5, dtype=np.float32).reshape(-1)
            measure_vector = [metric_method, accuracy_score, precision_score, recall_score, f1_score]
            for train_idx, test_idx in skf.split(embeds, Y):
                clf.fit(embeds[train_idx], Y[train_idx])
                y_pred, y_true = clf.predict(embeds[test_idx]), Y[test_idx]
                for k in range(5):
                    tmp[k] += measure_vector[k](y_true=y_true, y_pred=y_pred)
            tmp /= n_splits
            if self.verbose:
                Debugger.info_print('args {} for clf {}, performance: {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(
                    opt_args, self.kernel, tmp[1], tmp[2], tmp[3], tmp[4]))
            self.clf = {'clf': clf, 'clf-args': opt_args}
        self.clf['clf'].fit(X, Y)

    def predict(self, X, norm=False):
        assert self.shapelets is not None, 'shapelets has not been learnt yet...'
        assert self.clf is not 'classifier has not been learnt yet...'
        if norm:
            embeds = normalize(self.embed(x=X), axis=0)
        else:
            embeds = self.embed(x=X)
        return self.clf['clf'].predict(embeds)

    def save_model(self, fpath, **kwargs):
        pickle.dump(self.__dict__, open(fpath, 'wb'))

    def load_model(self, fpath, **kwargs):
        paras = pickle.load(open(fpath, 'rb'))
        for key, val in paras.items():
            self.__dict__[key] = val

    def save_shapelets(self, fpath):
        pickle.dump(self.shapelets, open(fpath, 'wb'))

    def load_shapelets(self, fpath):
        self.shapelets = pickle.load(open(fpath, 'rb'))
