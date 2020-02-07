# -*- coding: utf-8 -*-
import numpy as np


def greedy_dtw_path(x, y, warp, dist=lambda x, y: np.linalg.norm(x - y)):
    """
    generate dtw-path greedily.
    :param x:
    :param y:
    :param warp:
    :param dist:
    :return:
    """
    if np.ndim(x) == 1:
        x = x.reshape(-1, 1)
    if np.ndim(y) == 1:
        y = y.reshape(-1, 1)
    nrows, ncols = x.shape[0], y.shape[0]
    ridx, cidx, rpath, cpath = 0, 0, [0], [0]
    while ridx < nrows - 1 and cidx < ncols - 1:
        rdist = dist(x[ridx + 1], y[cidx])
        cdist = dist(x[ridx], y[cidx + 1])
        ddist = dist(x[ridx + 1], y[cidx + 1])
        if ddist < rdist and ddist < cdist:
            ridx += 1
            cidx += 1
        elif rdist < cdist:
            if ridx < cidx + warp:
                ridx += 1
            else:
                cidx += 1
        else:
            if cidx < ridx + warp:
                cidx += 1
            else:
                ridx += 1
        rpath.append(ridx)
        cpath.append(cidx)
    for k in range(ridx + 1, nrows):
        rpath.append(k)
        cpath.append(ncols - 1)
    for k in range(cidx + 1, ncols):
        cpath.append(k)
        rpath.append(nrows - 1)
    return np.array(rpath), np.array(cpath)


def parameterized_gdtw_npy(x, y, w, warp, dist=lambda x, y: np.linalg.norm(x - y)):
    if np.ndim(x) == 1:
        x = x.reshape(-1, 1)
    if np.ndim(y) == 1:
        y = y.reshape(-1, 1)
    dpath = greedy_dtw_path(x=x, y=y, dist=dist, warp=warp)
    return dist((x * np.abs(w).reshape(len(w), -1))[dpath[0]], y[dpath[1]])


def expand_array(y, warp):
    size = y.shape[0]
    tmp_y = np.concatenate((y[size - warp: size, :], y, y[: warp, :]), axis=0)
    return np.array([tmp_y[k: (k+2 * warp + 1)] for k in range(size)], dtype=np.float32)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


def softmax_1d(x):
    return np.exp(x) / np.sum(np.exp(x), keepdims=True)


def parameterized_gw_npy(x, y, w, warp):
    distance = np.sum((x.reshape(x.shape[0], -1, x.shape[1]) - expand_array(y=y, warp=warp)) ** 2,
                      axis=1)
    softmin_distance = np.sum(softmax(-distance.astype(np.float64)).astype(np.float32) * distance,
                              axis=1)
    return np.sqrt(np.sum(softmin_distance * np.abs(w)))


def pattern_distance_time_aware(pattern, time_series, local_factor, global_factor, warp,
                                init, measurement):
    """
    pattern distance with timing factors in numpy.
    :param pattern:
    :param time_series:
    :param local_factor:
    :param global_factor:
    :param warp:
    :param init:
    :param measurement:
    :return:
    """
    if measurement == 'gw':
        dist = parameterized_gw_npy
    elif measurement == 'gdtw':
        dist = parameterized_gdtw_npy
    else:
        raise NotImplementedError('unsupported distance {}'.format(measurement))
    num_segment = int(time_series.shape[0] / pattern.shape[0])
    seg_length = pattern.shape[0]
    assert init + num_segment <= len(global_factor)
    time_series = time_series.reshape(num_segment, seg_length, -1)
    ret = np.zeros(num_segment, np.float32).reshape(-1)
    for k in range(num_segment):
        ret[k] = dist(x=pattern, y=time_series[k], w=local_factor, warp=warp)
    return np.sum(softmax_1d(-ret * np.abs(global_factor[init: init + num_segment]))
                  * ret * np.abs(global_factor[init: init + num_segment]))


def pattern_distance_no_timing(pattern, time_series, warp, measurement):
    """
    pattern distance without timing factor in numpy.
    :param pattern:
    :param time_series:
    :param warp:
    :param measurement:
    :return:
    """
    if measurement == 'gw':
        dist = parameterized_gw_npy
    elif measurement == 'gdtw':
        dist = parameterized_gdtw_npy
    else:
        raise NotImplementedError('unsupported distance {}'.format(measurement))
    num_segment = int(time_series.shape[0] / pattern.shape[0])
    seg_length = pattern.shape[0]
    w = np.ones(seg_length, dtype=np.float32).reshape(-1)
    assert time_series.shape[0] == num_segment * pattern.shape[0]
    time_series = time_series.reshape(num_segment, pattern.shape[0], -1)
    ret = np.zeros(num_segment, np.float32).reshape(-1)
    for k in range(num_segment):
        ret[k] = dist(x=pattern, y=time_series[k], w=w, warp=warp)
    return np.sum(softmax(-ret) * ret)
