# -*- coding: utf-8 -*-
import argparse
import warnings
import os
from config import *
from time2graph.utils.base_utils import Debugger

if __name__ == '__main__':
    warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='stealing')
    parser.add_argument('--classpath', type=str,
                        default='{}/baselines/TimeSeriesClassification/'.format(module_path))
    parser.add_argument('--input', type=str, default='{}/dataset/'.format(module_path))
    parser.add_argument('--output', type=str, default='{}/dataset/'.format(module_path))
    parser.add_argument('--top', type=str, default='{}/baselines/TimeSeriesClassification/'
                                                   'out/production/TimeSeriesClassification'.format(module_path))
    parser.add_argument('--gpu_number', type=int, default=0)
    parser.add_argument('--clf', type=str, required=True)

    opt = parser.parse_args()
    all_clf = [
        'CID_DTW', 'DD_DTW', 'WDTW', 'ED', 'DTW',
        'LearnShapelets', 'FastShapelets', 'BagOfPatterns',
        'TSF', 'TSBF', 'LPS', 'ST', 'COTE'
    ]

    classpath = []
    for dirpath, dirnames, fnamesList in os.walk(opt.classpath):
        Debugger.info_print('{}'.format(dirpath))
        for fname in fnamesList:
            if fname.endswith('.jar'):
                classpath.append('{}{}'.format(dirpath, fname))
        break
    Debugger.info_print('{}'.format(classpath))

    cmd = 'CUDA_VISIBLE_DEVICES={} java -classpath {}'.format(opt.gpu_number, opt.top)
    if opt.clf != 'all':
        for p in classpath:
            cmd += ':{}'.format(p)
        dataset_cmd = cmd + ' development.DataSets -i {} -o {} -t {}'.format(opt.input, opt.output, opt.dataset)
        predict_cmd = cmd + ' timeseriesweka.examples.ClassificationExamples -i {} -o {} -t {} -c {}'.format(
            opt.input, opt.output, opt.dataset, opt.clf
        )
        output = open('{}/evaluate_baselines_{}_{}.sh'.format(opt.top, opt.dataset, opt.clf), 'w')
        output.write('#!/usr/bin/env bash\n{}\n{}\n'.format(dataset_cmd, predict_cmd))
        output.close()
    else:
        for p in classpath:
            cmd += ':{}'.format(p)
        dataset_cmd = cmd + ' development.DataSets -i {} -o {} -t {}'.format(opt.input, opt.output, opt.dataset)
        output = open('{}/evaluate_baselines_{}_{}.sh'.format(opt.top, opt.dataset, opt.clf), 'w')
        output.write('#!/usr/bin/env bash\n{}\n'.format(dataset_cmd))
        for clf in all_clf:
            predict_cmd = cmd + ' timeseriesweka.examples.ClassificationExamples -i {} -o {} -t {} -c {}'.format(
                opt.input, opt.output, opt.dataset, clf
            )
            output.write('{}\n'.format(predict_cmd))
        output.close()
