# -*- coding: utf-8 -*-
import argparse
import warnings
import os
from time2graph.utils.base_utils import Debugger

if __name__ == '__main__':
    warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='ucr-Earthquakes')
    parser.add_argument('--mode', type=str, default='embedding')
    parser.add_argument('--embed', type=str, default='aggregate')
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--paras', type=str, required=True)
    parser.add_argument('--gpu_number', type=int, default=0)
    parser.add_argument('--embed_size', type=int, default=256)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--K', type=int, default=100)
    parser.add_argument('--C', type=int, default=800)
    parser.add_argument('--num_segment', type=int, default=12)
    parser.add_argument('--seg_length', type=int, default=30)
    parser.add_argument('--total_length', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--warp', type=int, default=2)
    parser.add_argument('--njobs', type=int, default=20)
    parser.add_argument('--percentile', type=int, default=10)
    parser.add_argument('--init', type=int, default=0)

    opt = parser.parse_args()
    cmd = 'CUDA_VISIBLE_DEVICES={} python scripts/run.py --njobs {} ' \
          '--init {} --gpu_enable --dataset {} --mode embedding ' \
          '--percentile {} --batch_size {} --cmethod greedy --kernel xgb ' \
          '--embed {} --opt_metric accuracy'.format(
        opt.gpu_number, opt.njobs, opt.init, opt.dataset, opt.percentile, opt.batch_size, opt.embed)
    paras = {
        'K': opt.K,
        'seg_length': opt.seg_length,
        'num_segment': opt.num_segment,
        'embed_size': opt.embed_size,
        'warp': opt.warp
    }
    assert opt.target in paras
    if opt.target == 'seg_length':
        assert opt.total_length != -1
        paras.pop(opt.target)
        paras.pop('num_segment')
    else:
        paras.pop(opt.target)
    for key, val in paras.items():
        cmd += ' --{} {}'.format(key, val)

    output = open('evaluate_paras_{}.sh'.format(opt.target), 'w')
    output.write('#!/usr/bin/env bash\n')
    for p in opt.paras.split(','):
        if opt.target == 'K':
            tmp = '{} --{} {} --C {}'.format(cmd, opt.target, p, int(p) * 10)
            Debugger.info_print('running: {}'.format(tmp))
            output.write('{}\n'.format(tmp))
        elif opt.target == 'seg_length':
            tmp = '{} --{} {} --{} {} --C {}'.format(
                cmd, opt.target, p, 'num_segment',
                int(opt.total_length // int(p)), int(paras['K'] * 10))
            Debugger.info_print('running: {}'.format(tmp))
            output.write('{}\n'.format(tmp))
        else:
            tmp = '{} --{} {} --C {}'.format(cmd, opt.target, p, int(paras['K'] * 10))
            Debugger.info_print('running: {}'.format(tmp))
            output.write('{}\n'.format(tmp))
    os.system('chmod u+x evaluate_paras_{}.sh'.format(opt.target))
