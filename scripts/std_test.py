# -*- coding: utf-8 -*-
"""
    test scripts on three benchmark datasets: EQS, WTC, STB
"""
import argparse
import warnings
import os
from config import *
from archive.load_usr_dataset import load_usr_dataset_by_name
from time2graph.utils.base_utils import Debugger
from time2graph.core.model import Time2Graph
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


if __name__ == '__main__':
    warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='ucr-Earthquakes')
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--model_cache', action='store_true', default=False)
    parser.add_argument('--shapelet_cache', action='store_true', default=False)
    parser.add_argument('--gpu_enable', action='store_true', default=False)
    args = parser.parse_args()
    Debugger.info_print('running with {}'.format(args.__dict__))

    # set default options
    general_options = {
        'kernel': 'xgb',
        'opt_metric': 'accuracy',
        'init': 0,
        'warp': 2,
        'tflag': True,
        'mode': 'embedding',
        'candidate_method': 'greedy'
    }
    model_options = model_args[args.dataset]
    xgb_options = xgb_args[args.dataset]

    # load benchmark dataset
    if args.dataset.startswith('ucr'):
        dataset = args.dataset.rstrip('\n\r').split('-')[-1]
        x_train, y_train, x_test, y_test = load_usr_dataset_by_name(
            fname=dataset, length=model_options['seg_length'] * model_options['num_segment'])
    else:
        raise NotImplementedError()
    Debugger.info_print('training: {:.2f} positive ratio with {}'.format(
        float(sum(y_train) / len(y_train)), len(y_train)))
    Debugger.info_print('test: {:.2f} positive ratio with {}'.format(
        float(sum(y_test) / len(y_test)), len(y_test)))

    # initialize Time2Graph model
    m = Time2Graph(gpu_enable=args.gpu_enable, **model_options, **general_options,
                   shapelets_cache='{}/scripts/cache/{}_{}_{}_{}_shapelets.cache'.format(
                       module_path, args.dataset, general_options['candidate_method'],
                       model_options['K'], model_options['seg_length']))
    if args.model_cache:
        m.load_model(fpath='{}/scripts/cache/{}_embedding_t2g_model.cache'.format(module_path, args.dataset))
    if args.shapelet_cache:
        m.t2g.load_shapelets(fpath=m.shapelets_cache)
    res = np.zeros(4, dtype=np.float32)

    Debugger.info_print('training {}_tim2graph_model ...'.format(args.dataset))
    cache_dir = '{}/scripts/cache/{}'.format(module_path, args.dataset)

    if not path.isdir(cache_dir):
        os.mkdir(cache_dir)
    m.fit(X=x_train, Y=y_train, n_splits=args.n_splits, tuning=False, opt_args=xgb_options)
    y_pred = m.predict(X=x_test)[0]
    Debugger.info_print('classification result: accuracy {:.4f}, precision {:.4f}, recall {:.4f}, F1 {:.4f}'.format(
            accuracy_score(y_true=y_test, y_pred=y_pred),
            precision_score(y_true=y_test, y_pred=y_pred),
            recall_score(y_true=y_test, y_pred=y_pred),
            f1_score(y_true=y_test, y_pred=y_pred)
        ))
