# -*- coding: utf-8 -*-
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
    parser.add_argument('--K', type=int, default=100)
    parser.add_argument('--C', type=int, default=800)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--num_segment', type=int, default=12)
    parser.add_argument('--seg_length', type=int, default=30)
    parser.add_argument('--njobs', type=int, default=8)
    parser.add_argument('--data_size', type=int, default=1)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=0.05)
    parser.add_argument('--init', type=int, default=0)
    parser.add_argument('--gpu_enable', action='store_true', default=False)
    parser.add_argument('--opt_metric', type=str, default='accuracy')
    parser.add_argument('--cache', action='store_true', default=False)
    parser.add_argument('--embed', type=str, default='aggregate')
    parser.add_argument('--embed_size', type=int, default=256)
    parser.add_argument('--warp', type=int, default=2)
    parser.add_argument('--cmethod', type=str, default='greedy')
    parser.add_argument('--kernel', type=str, default='xgb')
    parser.add_argument('--percentile', type=int, default=10)
    parser.add_argument('--measurement', type=str, default='gdtw')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--tflag', action='store_false', default=True)
    parser.add_argument('--scaled', action='store_true', default=False)
    parser.add_argument('--norm', action='store_true', default=False)
    parser.add_argument('--no_global', action='store_false', default=True)

    args = parser.parse_args()
    Debugger.info_print('running with {}'.format(args.__dict__))

    if args.dataset.startswith('ucr'):
        dataset = args.dataset.rstrip('\n\r').split('-')[-1]
        x_train, y_train, x_test, y_test = load_usr_dataset_by_name(
            fname=dataset, length=args.seg_length * args.num_segment)
    else:
        raise NotImplementedError()
    Debugger.info_print('training: {:.2f} positive ratio with {}'.format(float(sum(y_train) / len(y_train)),
                                                                         len(y_train)))
    Debugger.info_print('test: {:.2f} positive ratio with {}'.format(float(sum(y_test) / len(y_test)),
                                                                     len(y_test)))
    m = Time2Graph(kernel=args.kernel, K=args.K, C=args.C, seg_length=args.seg_length,
                   opt_metric=args.opt_metric, init=args.init, gpu_enable=args.gpu_enable,
                   warp=args.warp, tflag=args.tflag, mode=args.embed,
                   percentile=args.percentile, candidate_method=args.cmethod,
                   batch_size=args.batch_size, njobs=args.njobs,
                   optimizer=args.optimizer, alpha=args.alpha,
                   beta=args.beta, measurement=args.measurement,
                   representation_size=args.embed_size, data_size=args.data_size,
                   scaled=args.scaled, norm=args.norm, global_flag=args.no_global,
                   shapelets_cache='{}/scripts/cache/{}_{}_{}_{}_shapelets.cache'.format(
                       module_path, args.dataset, args.cmethod, args.K, args.seg_length)
                   )

    res = np.zeros(4, dtype=np.float32)
    Debugger.info_print('training {}_mixed_model ...'.format(args.dataset))
    cache_dir = '{}/scripts/cache/{}/'.format(module_path, args.dataset)
    if not path.isdir(cache_dir):
        os.mkdir(cache_dir)
    m.fit(X=x_train, Y=y_train, cache_dir=cache_dir, n_splits=args.n_splits)
    if args.cache:
        m.save_model(fpath='{}/scripts/cache/{}_embedding_t2g_model.cache'.format(module_path, args.dataset))
    y_pred = m.predict(X=x_test)[0]
    Debugger.info_print('result: accu {:.4f}, prec {:.4f}, recall {:.4f}, f1 {:.4f}'.format(
            accuracy_score(y_true=y_test, y_pred=y_pred),
            precision_score(y_true=y_test, y_pred=y_pred),
            recall_score(y_true=y_test, y_pred=y_pred),
            f1_score(y_true=y_test, y_pred=y_pred)
        ))
