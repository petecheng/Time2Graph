# -*- coding: utf-8 -*-
import pickle
from config import *
from time2graph.utils.base_utils import ModelUtils
from time2graph.core.model_embeds import Time2GraphEmbed
from baselines.feature_based import FeatureModel
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Time2Graph(ModelUtils):
    """
        Main Class of Time2Graph Model.
    """
    def __init__(self, kernel, shapelets_cache, K, C, seg_length, init, opt_metric,
                 warp=2, tflag=True, gpu_enable=True, percentile=15, mode='concate',
                 batch_size=100, data_size=1, scaled=False, norm=False, **kwargs):
        """
        @param kernel:
            str, choice of outer-classifier; recommend using xgb, while valid candidates can be found in ModelUtils.
        @param shapelets_cache:
            str, the path of cache of shapelets.
        @param K:
            int, number of shapelets that try to learn.
        @param C:
            int, number of shapelet candidates when learning shapelets.
        @param seg_length:
            int, the length of a segment.
        @param init:
            int, initial offset in the original time series, default as 0.
        @param opt_metric:
            str, one of 'accuracy', 'precision', 'recall' and 'f1', on which to conduct fine-tuning.
        @param warp:
            int, warp step in greedy-dtw, default as 2.
        @param tflag:
            bool, flag that whether to add timing factors, default is True.
            That is it is set as False, it will learn static shapelets.
        @param gpu_enable:
            bool, whether to use gpu during computation.
        @param percentile:
            int, percentile that use to determine distance threshold when constructing shapelet evolution graph.
        @param mode:
            str, 'concate' or 'aggregate', the way to generate time series embeddings.
            That is, concate weighted segment embeddings or aggregate them as one.
        @param batch_size:
            int, batch size during training.
        @param data_size:
            int, the dimension of time series data,
            where we can denote time series shape as (N x L x data_size).
        @param scaled:
            bool, whether to scale time series by z-normalize
        @param norm:
            bool, whether to conduct min-max normalization when extract time series features
        @param kwargs:
            other candidate options, i.e.,
            model_cache: bool, whether to load model from cache
            other options in Time2GraphEmbed.
        """
        super(Time2Graph, self).__init__(kernel=kernel, **kwargs)
        self.shapelets_cache = shapelets_cache
        self.K = K
        self.C = C
        self.seg_length = seg_length
        self.init = init
        self.opt_metric = opt_metric
        self.warp = warp
        self.tflag = tflag
        self.gpu_enable = gpu_enable
        self.percentile = percentile
        self.mode = mode
        self.batch_size = batch_size
        self.data_size = data_size
        self.scaled = scaled
        self.norm = norm
        self.data_scaler = [StandardScaler() for _ in range(self.data_size)]
        self.feature_scaler = MinMaxScaler()
        model_cache = kwargs.get('model_cache', None)
        self.verbose = kwargs.get('verbose', False)
        if model_cache is not None:
            self.load_model(fpath=model_cache)
            Debugger.info_print('load time2graph model from cache {}...'.format(model_cache))
        else:
            self.t2g = Time2GraphEmbed(kernel=kernel, K=K, C=C, seg_length=seg_length,
                                       opt_metric=opt_metric, warp=warp, tflag=tflag,
                                       gpu_enable=gpu_enable, percentile=percentile, mode=mode,
                                       batch_size=batch_size, **kwargs)
            if path.isfile(self.shapelets_cache):
                self.t2g.load_shapelets(fpath=self.shapelets_cache)
            self.fm = FeatureModel(seg_length=self.t2g.seg_length, kernel=kernel)
            self.clf = self.clf__()

    def extract_features(self, X, init=0, train=False):
        """
        @param X:
            ndarray with shape (N x L x data_size), input time series
        @param init:
            int, the same as self.init
        @param train:
            bool, flag for training or not.
        @return:
            time series features (embeddings)
        """
        feat = self.fm.extract_features(samples=X)
        if self.scaled:
            X_scaled = np.zeros(X.shape, dtype=np.float)
            for k in range(self.data_size):
                X_scaled[:, :, k] = self.data_scaler[k].fit_transform(X[:, :, k])
            embed = self.t2g.embed(x=X_scaled, init=init)
        else:
            embed = self.t2g.embed(x=X, init=init)
        if self.norm:
            if train:
                feat = self.feature_scaler.fit_transform(X=feat)
            else:
                feat = self.feature_scaler.transform(X=feat)
        return np.concatenate((embed, feat), axis=1)

    def fit(self, X, Y, n_splits=5, balanced=True, cache_dir='{}/scripts/cache/'.format(module_path), **kwargs):
        """
        @param X:
            ndarray with shape (N x L x data_size), input time series.
        @param Y:
            ndarray with shape (N x 1), labels.
        @param n_splits:
            int, number of splits in cross-validation.
        @param balanced:
            bool, whether to balance the pos/neg during fitting classifier.
        @param cache_dir:
            str, cache dir for graph embeddings.
        @param kwargs:
            tuning: bool, whether to tune the parameters of outer-classifier(xgb).
            opt_args: dict, if tuning is False, opt_args must be given that
                the optimal parameters of outer-classifier should be pre-defined.
        """
        # fit data scaler
        for k in range(self.data_size):
            self.data_scaler[k].fit(X[:, :, k])
        X_scaled = np.zeros(X.shape, dtype=np.float)
        for k in range(self.data_size):
            X_scaled[:, :, k] = self.data_scaler[k].fit_transform(X[:, :, k])
        if self.t2g.shapelets is None:
            if self.scaled:
                self.t2g.learn_shapelets(
                    x=X_scaled, y=Y, num_segment=int(X_scaled.shape[1] / self.seg_length),
                    data_size=self.data_size, num_batch=int(X_scaled.shape[0] // self.batch_size))
            else:
                self.t2g.learn_shapelets(
                    x=X, y=Y, num_segment=int(X.shape[1] / self.seg_length),
                    data_size=self.data_size, num_batch=int(X.shape[0] // self.batch_size))
            self.t2g.save_shapelets(fpath=self.shapelets_cache)
            Debugger.info_print('saving shapelets cache to {}'.format(self.shapelets_cache))
        if self.t2g.sembeds is None:
            Debugger.info_print('training embedding model...')
            if self.scaled:
                self.t2g.fit_embedding_model(x=X_scaled, y=Y, cache_dir=cache_dir)
            else:
                self.t2g.fit_embedding_model(x=X, y=Y, cache_dir=cache_dir)
        x = self.extract_features(X=X, init=self.init)
        Debugger.info_print('extract mixed features done...')
        max_accu, max_prec, max_recall, max_f1, max_metric = -1, -1, -1, -1, -1
        metric_measure = self.return_metric_method(opt_metric=self.t2g.opt_metric)
        tuning, opt_args = kwargs.get('tuning', True), kwargs.get('opt_args', None)

        ###################################################
        # fine-tuning to find optimal classifier parameters
        if tuning:
            arguments = self.clf_paras(balanced=balanced)
            for args in arguments:
                self.clf.set_params(**args)
                skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
                tmp = np.zeros(5, dtype=np.float32).reshape(-1)
                measure_vector = [metric_measure, accuracy_score, precision_score, recall_score, f1_score]
                for train_idx, test_idx in skf.split(x, Y):
                    self.clf.fit(x[train_idx], Y[train_idx])
                    y_pred, y_true = self.clf.predict(x[test_idx]), Y[test_idx]
                    for k in range(5):
                        tmp[k] += measure_vector[k](y_true=y_true, y_pred=y_pred)
                tmp /= n_splits
                Debugger.debug_print('args tuning: accu {:.4f}, prec {:.4f}, recall {:.4f}, f1 {:.4f}'.format(
                    tmp[1], tmp[2], tmp[3], tmp[4]
                ), debug=self.verbose)
                if max_metric < tmp[0]:
                    max_metric = tmp[0]
                    opt_args = args
                    max_accu, max_prec, max_recall, max_f1 = tmp[1:]
            if self.verbose:
                Debugger.info_print('args {} for clf {}, performance: {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(
                    opt_args, self.kernel, max_accu, max_prec, max_recall, max_f1))
            self.clf.set_params(**opt_args)

        ###################################################
        # load optimal parameters predefined before.
        else:
            assert opt_args is not None, 'missing opt args specified'
            self.clf.set_params(**opt_args)
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
            tmp = np.zeros(5, dtype=np.float32).reshape(-1)
            measure_vector = [metric_measure, accuracy_score, precision_score, recall_score, f1_score]
            for train_idx, test_idx in skf.split(x, Y):
                self.clf.fit(x[train_idx], Y[train_idx])
                y_pred, y_true = self.clf.predict(x[test_idx]), Y[test_idx]
                for k in range(5):
                    tmp[k] += measure_vector[k](y_true=y_true, y_pred=y_pred)
            tmp /= n_splits
            if self.verbose:
                Debugger.info_print('args {} for clf {}, performance: {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(
                    opt_args, self.kernel, tmp[1], tmp[2], tmp[3], tmp[4]))
        self.clf.fit(x, Y)

    def predict(self, X, **kwargs):
        """
        :param X:
            input, with shape [N, T, data_size].
        :param kwargs:
            ignore.
        :return:
            predicted label, predicted probability.
        """
        x = self.extract_features(X=X, init=self.init)
        return self.clf.predict(x), self.clf.predict_proba(x)[:, 1]

    def save_model(self, fpath, **kwargs):
        """
        dump model to a specific path.
        :param fpath:
            saving path.
        :param kwargs:
            ignore.
        :return:
        """
        pickle.dump(self.__dict__, open(fpath, 'wb'))

    def load_model(self, fpath, **kwargs):
        """
        save model from a given cache file.
        :param fpath:
            loading path.
        :param kwargs:
            ignore.
        :return:
        """
        paras = pickle.load(open(fpath, 'rb'))
        for key, val in paras.items():
            self.__dict__[key] = val
