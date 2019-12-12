# -*- coding: utf-8 -*-
import numpy as np
from os import path
from time2graph.utils.base_utils import Debugger

module_path = path.dirname(path.abspath(__file__))
EQS = {
    'K': 50,
    'C': 800,
    'seg_length': 24,
    'num_segment': 21,
    'percentile': 5
}

WTC = {
    'K': 20,
    'C': 400,
    'seg_length': 30,
    'num_segment': 30,
    'percentile': 5,
    'no_global': True
}

STB = {
    'K': 50,
    'C': 800,
    'seg_length': 15,
    'num_segment': 15,
    'percentile': 10,
    'embed': 'aggregate'
}

model_args = {
    'ucr-Earthquakes': EQS,
    'ucr-WormsTwoClass': WTC,
    'ucr-Strawberry': STB
}

xgb_args = {
    'ucr-Earthquakes': {
        'max_depth': 16,
        'learning_rate': 0.2,
        'scale_pos_weight': 1,
        'booster': 'gbtree'
    },
    'ucr-WormsTwoClass': {
        'max_depth': 2,
        'learning_rate': 0.2,
        'scale_pos_weight': 1,
        'booster': 'gbtree'
    },
    'ucr-Strawberry': {
        'max_depth': 8,
        'learning_rate': 0.2,
        'scale_pos_weight': 1,
        'booster': 'gbtree'
    }
}

__all__ = [
    'np',
    'path',
    'Debugger',
    'module_path',
    'model_args',
    'xgb_args'
]
