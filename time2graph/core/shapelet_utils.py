# -*- coding: utf-8 -*-
from sklearn.cluster import KMeans
from sklearn.preprocessing import minmax_scale
from .distance_utils import *
from ..utils.base_utils import Debugger, syscmd
from ..utils.mp_utils import ParMap, parallel_monitor, NJOBS
__tmat_threshold = 1e-2


def softmax_np(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def __candidate_cluster_factory(n_clusters, seg_length):
    def __main__(pid, args, queue):
        ret = []
        for time_series_segments in args:
            kmeans = KMeans(n_clusters=n_clusters).fit(time_series_segments)
            ret.append(kmeans.cluster_centers_.reshape(n_clusters, seg_length, -1))
            queue.put(0)
        return ret
    return __main__


def __candidate_greedy_factory(n_candiates, seg_length):
    def __main__(pid, args, queue):
        ret = []
        for time_series_segments in args:
            size = time_series_segments.shape[0]
            center_segment = np.mean(time_series_segments, axis=0)
            cand_dist = np.linalg.norm(
                time_series_segments.reshape(size, -1) - center_segment.reshape(1, -1), axis=1)
            tmp = []
            for cnt in range(n_candiates):
                idx = np.argmax(cand_dist)
                cand_dist[idx] = -1
                update_idx = cand_dist >= 0
                dims = np.sum(update_idx)
                cand_dist[update_idx] += np.linalg.norm(
                    time_series_segments[update_idx].reshape(dims, -1) - time_series_segments[idx].reshape(1, -1),
                    axis=1
                )
                tmp.append(time_series_segments[idx].reshape(seg_length, -1))
            ret.append(tmp)
            queue.put(0)
        return ret
    return __main__


def generate_shapelet_candidate(time_series_set, num_segment, seg_length, candidate_size, **kwargs):
    __method, __debug = kwargs.get('candidate_method', 'greedy'), kwargs.get('debug', True)
    njobs = kwargs.get('njobs', NJOBS)
    Debugger.debug_print('begin to generate shapelet candidates...', __debug)
    num_time_series = time_series_set.shape[0]
    time_series_set = time_series_set.reshape(num_time_series, num_segment, seg_length, -1)
    assert candidate_size >= num_segment, 'candidate-size {} should be larger ' \
                                          'than n_segments {}'.format(candidate_size, num_segment)
    args, n_clusters = [], candidate_size // num_segment
    for idx in range(num_segment):
        args.append(time_series_set[:, idx, :, :].reshape(num_time_series, -1))
    if __method == 'cluster':
        work_func = __candidate_cluster_factory
    elif __method == 'greedy':
        work_func = __candidate_greedy_factory
    else:
        raise NotImplementedError('unsupported candidate generating method {}'.format(__method))
    parmap = ParMap(
        work=work_func(n_clusters, seg_length),
        monitor=parallel_monitor(msg='generate candidate by {}'.format(__method),
                                 size=num_segment, debug=__debug),
        njobs=njobs
    )
    ret = np.concatenate(parmap.run(data=args), axis=0)
    Debugger.info_print('candidates with length {} sampling done...'.format(seg_length))
    Debugger.info_print('totally {} candidates with shape {}'.format(len(ret), ret.shape))
    return ret


def __shapelet_distance_factory(shapelets, num_segment, seg_length, tflag,
                                init, warp, dist, global_flag):
    def __main__(pid, args, queue):
        ret = []
        for time_series in args:
            time_series = time_series.reshape(num_segment, seg_length, -1)
            tmp = np.zeros((num_segment, len(shapelets)), dtype=np.float32)
            if tflag and global_flag:
                for idx, (pattern, local_factor, global_factor, _) in enumerate(shapelets):
                    for k in range(num_segment):
                        tmp[k, idx] = dist(x=pattern, y=time_series[k],
                                           w=local_factor, warp=warp) * np.abs(global_factor[init + k])
            elif tflag and not global_flag:
                for idx, (pattern, local_factor, global_factor, _) in enumerate(shapelets):
                    for k in range(num_segment):
                        tmp[k, idx] = dist(x=pattern, y=time_series[k], w=local_factor, warp=warp)
            else:
                for idx, (pattern, _) in enumerate(shapelets):
                    for k in range(num_segment):
                        tmp[k, idx] = dist(x=pattern, y=time_series[k],
                                           w=np.ones(pattern.shape[0]), warp=warp)
            ret.append(tmp)
            queue.put(0)
        return ret
    return __main__


def shapelet_distance(time_series_set, shapelets, seg_length, tflag, tanh, debug, init,
                      warp, measurement, global_flag):
    num_time_series = time_series_set.shape[0]
    num_segment = int(time_series_set.shape[1] / seg_length)
    num_shapelet = len(shapelets)
    assert num_segment * seg_length == time_series_set.shape[1]
    if measurement == 'gw':
        dist = parameterized_gw_npy
    elif measurement == 'gdtw':
        dist = parameterized_gdtw_npy
    else:
        raise NotImplementedError('unsupported distance {}'.format(measurement))
    parmap = ParMap(
        work=__shapelet_distance_factory(
            shapelets=shapelets, num_segment=num_segment, seg_length=seg_length,
            tflag=tflag, init=init, warp=warp, dist=dist, global_flag=global_flag),
        monitor=parallel_monitor(msg='shapelet distance', size=num_time_series, debug=debug),
        njobs=NJOBS
    )
    sdist = np.array(parmap.run(data=list(time_series_set)), dtype=np.float32).reshape(
        time_series_set.shape[0], num_segment, num_shapelet
    )
    if tanh:
        sdist = np.tanh(sdist)
    return sdist


def transition_matrix(time_series_set, shapelets, seg_length, tflag, multi_graph,
                      percentile, threshold, tanh, debug, init, warp, measurement, global_flag):
    num_time_series = time_series_set.shape[0]
    num_segment = int(time_series_set.shape[1] / seg_length)
    num_shapelet = len(shapelets)
    if multi_graph:
        gcnt = num_segment - 1
    else:
        gcnt = 1
    tmat = np.zeros((gcnt, num_shapelet, num_shapelet), dtype=np.float32)
    sdist = shapelet_distance(
        time_series_set=time_series_set, shapelets=shapelets, seg_length=seg_length, tflag=tflag,
        tanh=tanh, debug=debug, init=init, warp=warp, measurement=measurement, global_flag=global_flag
    )
    if percentile is not None:
        dist_threshold = np.percentile(sdist, percentile)
        Debugger.info_print('threshold({}) {}, mean {}'.format(percentile, dist_threshold, np.mean(sdist)))
    else:
        dist_threshold = threshold
        Debugger.info_print('threshold {}, mean {}'.format(dist_threshold, np.mean(sdist)))

    n_edges = 0
    for tidx in range(num_time_series):
        for sidx in range(num_segment - 1):
            src_dist = sdist[tidx, sidx, :]
            dst_dist = sdist[tidx, sidx + 1, :]
            src_idx = np.argwhere(src_dist <= dist_threshold).reshape(-1)
            dst_idx = np.argwhere(dst_dist <= dist_threshold).reshape(-1)
            if len(src_idx) == 0 or len(dst_idx) == 0:
                continue
            n_edges += len(src_idx) * len(dst_idx)
            src_dist[src_idx] = 1.0 - minmax_scale(src_dist[src_idx])
            dst_dist[dst_idx] = 1.0 - minmax_scale(dst_dist[dst_idx])
            for src in src_idx:
                if multi_graph:
                    tmat[sidx, src, dst_idx] += (src_dist[src] * dst_dist[dst_idx])
                else:
                    tmat[0, src, dst_idx] += (src_dist[src] * dst_dist[dst_idx])
        Debugger.debug_print(
            '{:.2f}% transition matrix computed...'.format(float(tidx + 1) * 100 / num_time_series),
            debug=debug
        )
    Debugger.info_print('{} edges involved in shapelets graph'.format(n_edges))
    tmat[tmat <= __tmat_threshold] = 0.0
    for k in range(gcnt):
        for i in range(num_shapelet):
            norms = np.sum(tmat[k, i, :])
            if norms == 0:
                tmat[k, i, i] = 1.0
            else:
                tmat[k, i, :] /= np.sum(tmat[k, i, :])
    return tmat, sdist, dist_threshold


def __mat2edgelist(tmat, fpath):
    mat_shape = tmat.shape
    with open(fpath, 'w') as f:
        for src in range(mat_shape[0]):
            flag = False
            for dst in range(mat_shape[1]):
                if tmat[src, dst] <= 1e-5:
                    continue
                f.write('{} {}  {:.5f}\n'.format(src, dst, tmat[src, dst]))
                flag = True
            if not flag:
                f.write('{} {}  1.0000\n'.format(src, src))
        f.close()


def __embedding2mat(fpath, num_vertices, embed_size):
    mat = np.zeros((num_vertices, embed_size), dtype=np.float32)
    with open(fpath, 'r') as f:
        cnt = -1
        for line in f:
            if cnt < 0:
                cnt += 1
                continue
            line = line.split(' ')
            idx = int(line[0])
            for k in range(embed_size):
                mat[idx, k] = float(line[k + 1])
        f.close()
    return mat


def graph_embedding(tmat, num_shapelet, embed_size, cache_dir, **deepwalk_paras):
    __deepwalk_args__ = []
    Debugger.info_print('embed_size: {}'.format(embed_size))
    ret = []
    Debugger.info_print('transition matrix size {}'.format(tmat.shape))
    for idx in range(tmat.shape[0]):
        edgelist_path = '{}/{}.edgelist'.format(cache_dir, idx)
        embedding_path = '{}/{}.embeddings'.format(cache_dir, idx)
        __mat2edgelist(tmat=tmat[idx, :, :], fpath=edgelist_path)
        deepwalk_cmd = [
            'deepwalk --input {} --format weighted_edgelist --output {} --representation-size {}'.format(
                edgelist_path, embedding_path, embed_size)
        ]
        for key, val in deepwalk_paras.items():
            if key in __deepwalk_args__:
                deepwalk_cmd.append('--{} {}'.format(key, val))
        deepwalk_cmd = ' '.join(deepwalk_cmd)
        Debugger.info_print('run deepwalk with: {}'.format(deepwalk_cmd))
        _ = syscmd(deepwalk_cmd)
        ret.append(__embedding2mat(fpath=embedding_path, num_vertices=num_shapelet,
                                   embed_size=embed_size))
    return np.array(ret, dtype=np.float32).reshape(tmat.shape[0], num_shapelet, embed_size)
