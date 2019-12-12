# -*- coding: utf-8 -*-
from __future__ import print_function
import itertools
import dill
import contextlib
import math
import multiprocessing as mp
import numpy as np
from .base_utils import Debugger

NJOBS = mp.cpu_count()
if NJOBS >= 20:
    NJOBS = 20

__all__ = [
    'NJOBS',
    'ParMap',
    'parallel_monitor'
]


class ParMap(object):
    def __init__(self, work, monitor=None, njobs=NJOBS, maxtasksperchild=100):
        self.work_func = work
        self.monitor_func = monitor
        self.__njobs = njobs
        self.__mtpc = maxtasksperchild

        self.__pool = None

    def close(self):
        if self.__pool is not None:
            self.__pool.close()
            self.__pool.join()
        self.__pool = None

    def __del__(self):
        self.close()

    @property
    def njobs(self):
        return self.__njobs

    @njobs.setter
    def njobs(self, n):
        self.__njobs = n
        self.close()

    def default_chunk(self, dlen):
        return int(math.ceil(float(dlen) / self.njobs))

    def run(self, data, chunk=None, shuffle=False):
        if chunk is None:
            chunk = self.default_chunk(len(data))

        if shuffle:
            data, order, invorder = shuffle_sample(data)
        else:
            invorder = None

        slices = slice_sample(data, chunk=chunk)
        res = self.run_slices(slices)

        if shuffle:
            res = apply_order(res, invorder)

        return res

    def run_slices(self, slices):
        mgr = mp.Manager()
        report_queue = mgr.Queue()
        if self.monitor_func is not None:
            monitor = mp.Process(target=self.monitor_func, args=(report_queue,))
            monitor.start()
        else:
            monitor = None

        if self.njobs == 1:
            res = []
            for slc in slices:
                res.append(self.work_func(None, slc, report_queue))
        else:
            dill_work_func = dill.dumps(self.work_func)
            with contextlib.closing(mp.Pool(self.njobs, maxtasksperchild=self.__mtpc)) as pool:
                res = pool.map(func_wrapper, [[dill_work_func, slc, report_queue] for slc in slices])
        res = list(itertools.chain.from_iterable(res))

        report_queue.put(StopIteration())
        if monitor is not None:
            monitor.join()

        return res


def func_wrapper(args):
    func = dill.loads(args[0])
    return func(mp.current_process().ident, *args[1:])


def apply_order(sample, order):
    return [sample[o] for o in order]


def shuffle_sample(sample):
    order = np.random.permutation(np.arange(len(sample)))
    invorder = np.zeros((len(sample), ), dtype='int32')
    invorder[order] = np.arange(len(sample))

    return apply_order(sample, order), order, invorder


def slice_sample(sample, chunk=None, nslice=None):
    slices = []
    if chunk is None:
        chunk = int(len(sample) / nslice)
    else:
        if nslice is not None:
            raise RuntimeError("chunk ({}) and slice ({}) should not be specified simultaneously".format(chunk, nslice))

    curstart = 0
    while True:
        if curstart >= len(sample):
            break
        slices.append(sample[curstart:min(curstart + chunk, len(sample))])
        curstart += chunk

    return slices


def parallel_monitor(msg, size, debug):
    def monitor(queue):
        cnt = 0
        while True:
            obj = queue.get()
            if isinstance(obj, StopIteration):
                break
            if isinstance(obj, int):
                if obj != 0:
                    cnt += obj
                else:
                    cnt += 1
            else:
                cnt += 1
            Debugger.debug_print(msg='{} executed by {:.2f}%'.format(msg, float(cnt) / size * 100),
                                 debug=debug)
    return monitor
