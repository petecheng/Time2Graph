# -*- coding: utf-8 -*-
from config import *
from time2graph.utils.base_utils import ModelUtils
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class FeatureModel(ModelUtils):
    """
        Class for Handcraft-feature Model for time series classification.
        Feature list:
            a) mean, std of whole time series.
            b) mean, std of each segments.
            c) mean of the std of segments.
            d) std of the mean of segments.
    """
    def __init__(self, seg_length, kernel='xgb', opt_metric='f1', **kwargs):
        super(FeatureModel, self).__init__(kernel=kernel, **kwargs)
        self.clf = None
        self.seg_length = seg_length
        self.opt_metric = opt_metric

    def extract_features(self, samples):
        num_samples, data_size = samples.shape[0], samples.shape[-1]
        samples = samples.reshape(num_samples, -1, self.seg_length, data_size)
        series_mean = np.mean(samples.reshape(num_samples, -1, data_size), axis=1).reshape(num_samples, -1)
        series_std = np.std(samples.reshape(num_samples, -1, data_size), axis=1).reshape(num_samples, -1)
        seg_mean, seg_std = np.mean(samples, axis=2), np.mean(samples, axis=2)
        seg_mean_std, seg_std_mean = np.std(seg_mean, axis=1), np.mean(seg_std, axis=1)
        seg_mean = seg_mean.reshape(num_samples, -1)
        seg_std = seg_std.reshape(num_samples, -1)
        seg_mean_std = seg_mean_std.reshape(num_samples, -1)
        seg_std_mean = seg_std_mean.reshape(num_samples, -1)
        return np.concatenate((series_mean, series_std, seg_mean, seg_std, seg_mean_std, seg_std_mean), axis=1)

    def fit(self, X, Y, n_splits=5, balanced=True):
        x = self.extract_features(samples=X)
        max_accu, max_prec, max_recall, max_f1, max_metric = -1, -1, -1, -1, -1
        arguments, opt_args = self.clf_paras(balanced=balanced), None
        metric_measure = self.return_metric_method(opt_metric=self.opt_metric)
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
            if max_metric < tmp[0]:
                max_metric = tmp
                opt_args = args
                max_accu, max_prec, max_recall, max_f1 = tmp[1:]
        Debugger.info_print('args {} for clf {}, performance: {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(
            opt_args, self.kernel, max_accu, max_prec, max_recall, max_f1))
        self.clf.set_params(**opt_args)

    def predict(self, X, **kwargs):
        x = self.extract_features(samples=X)
        return self.clf.predict(x)
