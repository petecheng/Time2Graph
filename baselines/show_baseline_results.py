# -*- coding: utf-8 -*-
import argparse
import warnings
from config import *
from time2graph.utils.base_utils import Debugger
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def load_baseline_results(fpath):
    y_pred, y_test = [], []
    with open(fpath, 'r') as f:
        cnt = 0
        for line in f:
            if cnt < 3:
                cnt += 1
                continue
            line = line.rstrip('\n').split(',')
            if len(line) <= 4:
                continue
            y_test.append(int(line[0]))
            y_pred.append(int(line[1]))
        f.close()
    return y_pred, y_test


if __name__ == '__main__':
    warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='stealing')
    parser.add_argument('--clf', type=str, required=True)

    opt = parser.parse_args()
    all_clf = [
        'CID_DTW', 'DD_DTW', 'WDTW', 'ED', 'DTW',
        'LearnShapelets', 'FastShapelets', 'BagOfPatterns',
        'TSF', 'TSBF', 'LPS', 'SAX', 'ST', 'COTE', 'EE'
    ]
    assert opt.clf in all_clf
    fpath = '{}/dataset/{}/Predictions/{}/testFold0.csv'.format(module_path, opt.clf, opt.dataset)
    y_pred, y_test = load_baseline_results(fpath=fpath)
    Debugger.info_print('{} test samples with {:.4f} positive'.format(len(y_test), sum(y_test) / len(y_test)))
    accu = accuracy_score(y_true=y_test, y_pred=y_pred)
    prec = precision_score(y_true=y_test, y_pred=y_pred)
    recall = recall_score(y_true=y_test, y_pred=y_pred)
    f1 = f1_score(y_true=y_test, y_pred=y_pred)
    Debugger.info_print('res: accu {:.4f}, prec {:.4f}, recall {:.4f}, f1 {:.4f}'.format(
        accu, prec, recall, f1
    ))
