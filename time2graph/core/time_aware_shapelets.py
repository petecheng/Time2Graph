# -*- coding: utf-8 -*-
import torch
from torch.autograd import *
from torch import optim
from torch.nn import functional as F
from torch.distributions.normal import Normal
from torch.utils.data import DataLoader
from ..utils.base_utils import Queue
from .model_utils import *
from .shapelet_utils import *
from .distance_utils import *


def parameterized_gw_torch(x, y, w, torch_dtype, warp=2):
    """
    gw distance in torch with timing factors.
    :param x:
    :param y:
    :param w:
    :param torch_dtype:
    :param warp:
    :return:
    """
    distance = np.sum((x.reshape(x.shape[0], -1, x.shape[1]) - expand_array(y=y, warp=warp)) ** 2,
                      axis=1)
    assert not torch.any(torch.isnan(w)), 'local: {}'.format(w)
    softmin_distance = np.sum(softmax(-distance.astype(np.float64)).astype(np.float32) * distance,
                              axis=1)
    return torch.sqrt(torch.sum(torch.from_numpy(softmin_distance).type(torch_dtype) * torch.abs(w)))


def parameterized_gdtw_torch(x, y, w, torch_dtype, warp=2):
    """
    greedy-dtw distance in torch with timing factors.
    :param x:
    :param y:
    :param w:
    :param torch_dtype:
    :param warp:
    :return:
    """
    dpath = greedy_dtw_path(x=x, y=y, warp=warp)
    return torch.norm((torch.from_numpy(x).type(torch_dtype) * w.reshape(x.shape[0], -1))[dpath[0]] -
                      torch.from_numpy(y[dpath[1]]).type(torch_dtype))


def pattern_distance_torch(pattern, time_series, num_segment, seg_length,
                           local_factor, global_factor, torch_dtype, measurement):
    """
    compute distances between a pattern and a given time series.
    :param pattern:
    :param time_series:
    :param num_segment:
    :param seg_length:
    :param local_factor:
    :param global_factor:
    :param torch_dtype:
    :param measurement:
    :return:
    """
    if measurement == 'gw':
        dist_torch = parameterized_gw_torch
    elif measurement == 'gdtw':
        dist_torch = parameterized_gdtw_torch
    else:
        raise NotImplementedError('unsupported distance {}'.format(measurement))
    assert isinstance(time_series, np.ndarray) and isinstance(pattern, np.ndarray)
    time_series = time_series.reshape(num_segment, seg_length, -1)
    distance = Variable(torch.zeros(num_segment)).type(torch_dtype)
    for k in range(num_segment):
        distance[k] = dist_torch(x=pattern, y=time_series[k], w=local_factor, torch_dtype=torch_dtype)
    return torch.sum(F.softmax(-distance * torch.abs(global_factor), dim=0)
                     * (distance * torch.abs(global_factor)))


def __shapelet_candidate_loss(cand, time_series_set, label, num_segment, seg_length,
                              data_size, p, lr, alpha, beta, num_batch, gpu_enable,
                              measurement, **kwargs):
    """
    loss for learning time-aware shapelets.
    :param cand:
    :param time_series_set:
    :param label:
    :param num_segment:
    :param seg_length:
    :param data_size:
    :param p:
        normalizing parameter (0, 1, or 2).
    :param lr:
        learning rate.
    :param alpha:
        penalty weight for local timing factor.
    :param beta:
        penalty weight for global timing factor.
    :param num_batch:
    :param gpu_enable:
    :param measurement:
    :param kwargs:
    :return:
    """
    if gpu_enable:
        torch_dtype = torch.cuda.FloatTensor
    else:
        torch_dtype = torch.FloatTensor
    dataset_numpy = NumpyDataset(time_series_set, label)
    num_class = len(np.unique(label).reshape(-1))
    batch_size = int(len(dataset_numpy) // num_batch)
    local_factor_variable = Variable(torch.ones(seg_length).type(torch_dtype) / seg_length, requires_grad=True)
    global_factor_variable = Variable(torch.ones(num_segment).type(torch_dtype) / num_segment, requires_grad=True)
    current_loss, loss_queue, cnt, nan_cnt = 0.0, Queue(max_size=int(num_batch * 0.2)), 0, 0
    current_main_loss, current_penalty_loss = 0.0, 0.0
    max_iters, optimizer = kwargs.get('max_iters', 1), kwargs.get('optimizer', 'Adam')
    if optimizer == 'Adam':
        optimizer = optim.Adam
    elif optimizer == 'Adadelta':
        optimizer = optim.Adadelta
    elif optimizer == 'Adamax':
        optimizer = optim.Adamax
    else:
        raise NotImplementedError('unsupported optimizer {} for time-aware shapelets learning'.format(optimizer))
    optimizer = optimizer([local_factor_variable, global_factor_variable], lr=lr)

    while cnt < max_iters:
        sampler = StratifiedSampler(label=label, num_class=num_class)
        dataloader = DataLoader(dataset=dataset_numpy, batch_size=batch_size, sampler=sampler)
        batch_cnt = 0
        for x, y in dataloader:
            x = np.array(x, dtype=np.float32).reshape(len(x), -1, data_size)
            y = np.array(y, dtype=np.float32).reshape(-1)
            assert not np.any(np.isnan(x)), 'original time series data with nan'
            lb_idx, sample_flag = [], True
            for k in range(num_class):
                tmp_idx = np.argwhere(y == k).reshape(-1)
                if k >= 1 and len(tmp_idx) > 0:
                    sample_flag = False
                lb_idx.append(tmp_idx)
            if len(lb_idx[0]) == 0 or sample_flag:
                Debugger.debug_print('weighted sampling exception, positive {:.2f}/{}'.format(np.sum(y)/len(y), len(y)))
                continue
            loss = torch.Tensor([0.0]).type(torch_dtype)
            main_loss = torch.Tensor([0.0]).type(torch_dtype)
            penalty_loss = torch.Tensor([0.0]).type(torch_dtype)
            dist_tensor = torch.zeros(x.shape[0]).type(torch_dtype)
            for k in range(x.shape[0]):
                dist_tensor[k] = pattern_distance_torch(
                    pattern=cand, time_series=x[k, :, :], num_segment=num_segment,
                    seg_length=seg_length, local_factor=local_factor_variable,
                    global_factor=global_factor_variable, torch_dtype=torch_dtype,
                    measurement=measurement
                    # ignore the warning of reshape/view for local_factor_variable
                )
            assert not torch.isnan(dist_tensor).any(), 'dist: {}\nlocal: {}\nglobal: {}'.format(
                dist_tensor, local_factor_variable, global_factor_variable)
            mean, std = torch.mean(dist_tensor), torch.std(dist_tensor)
            dist_tensor = (dist_tensor - mean) / std
            # Debugger.info_print('transform: {}, {}'.format(torch.max(dist_tensor), torch.min(dist_tensor)))
            # Debugger.time_print(msg='pattern distance', begin=begin, profiling=True)
            for k in range(1, len(lb_idx)):
                src = dist_tensor[lb_idx[0]]
                dst = dist_tensor[lb_idx[k]]
                loss -= torch.abs(torch.distributions.kl.kl_divergence(
                    Normal(torch.mean(src), torch.std(src)),
                    Normal(torch.mean(dst), torch.std(dst))))
                main_loss -= torch.abs(torch.distributions.kl.kl_divergence(
                    Normal(torch.mean(src), torch.std(src)),
                    Normal(torch.mean(dst), torch.std(dst))))
                # Debugger.info_print('KL-loss: {}'.format(loss))
            loss += (alpha * torch.norm(local_factor_variable, p=p) / seg_length)
            loss += (beta * torch.norm(global_factor_variable, p=p) / num_segment)

            penalty_loss += (alpha * torch.norm(local_factor_variable, p=p) / seg_length)
            penalty_loss += (beta * torch.norm(global_factor_variable, p=p) / num_segment)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if gpu_enable:
                current_loss = float(loss.cpu().data.numpy())
                current_main_loss = float(main_loss.cpu().data)
                current_penalty_loss = float(penalty_loss.cpu().data)
            else:
                current_loss = float(loss.data.numpy())
                current_main_loss = float(main_loss.data)
                current_penalty_loss = float(penalty_loss.data)
            loss_queue.enqueue(current_loss)
            if np.isnan(current_loss) or torch.any(torch.isnan(local_factor_variable))\
                    or torch.any(torch.isnan(global_factor_variable)):
                local_factor_variable = Variable(torch.ones(seg_length).type(torch_dtype) / seg_length, requires_grad=True)
                global_factor_variable = Variable(torch.ones(num_segment).type(torch_dtype) / num_segment, requires_grad=True)
                current_loss = 1e5
                nan_cnt += 1
                if nan_cnt >= max_iters:
                    break
            else:
                Debugger.debug_print('{:.2f}% steps, loss {:.6f} with {:.6f} and penalty {:.6f}'.format(
                    batch_cnt * 100 / num_batch, current_loss, current_main_loss, current_penalty_loss))
            batch_cnt += 1
        cnt += 1
        if nan_cnt >= max_iters:
            break
        else:
            avg_loss = np.mean(loss_queue.queue[1:])
            if abs(current_loss - avg_loss) < kwargs.get('epsilon', 1e-2):
                break
    local_factor_variable = torch.abs(local_factor_variable)
    global_factor_variable = torch.abs(global_factor_variable)
    if gpu_enable:
        local_factor = local_factor_variable.cpu().data.numpy()
        global_factor = global_factor_variable.cpu().data.numpy()
    else:
        local_factor = local_factor_variable.data.numpy()
        global_factor = global_factor_variable.data.numpy()
    return local_factor, global_factor, current_loss, current_main_loss, current_penalty_loss


def __shapelet_candidate_loss_factory(time_series_set, label, num_segment,
                                      seg_length, data_size, p, lr, alpha, beta, num_batch,
                                      gpu_enable, measurement, **kwargs):
    """
    paralleling compute shapelet losses.
    :param time_series_set:
    :param label:
    :param num_segment:
    :param seg_length:
    :param data_size:
    :param p:
    :param lr:
    :param alpha:
    :param beta:
    :param num_batch:
    :param gpu_enable:
    :param measurement:
    :param kwargs:
    :return:
    """
    def __main__(pid, args, queue):
        ret = []
        for cand in args:
            local_factor, global_factor, loss, main_loss, penalty = __shapelet_candidate_loss(
                cand=cand, time_series_set=time_series_set, label=label, num_segment=num_segment,
                seg_length=seg_length, data_size=data_size, p=p, lr=lr,
                alpha=alpha, beta=beta, num_batch=num_batch, gpu_enable=gpu_enable,
                measurement=measurement, **kwargs
            )
            ret.append((cand, local_factor, global_factor, loss, main_loss, penalty))
            queue.put(0)
        return ret
    return __main__


def learn_time_aware_shapelets(time_series_set, label, K, C, num_segment, seg_length, data_size,
                               p, lr, alpha, beta, num_batch, gpu_enable, measurement, **kwargs):
    """
    learn time-aware shapelets.
    :param time_series_set:
        input time series data.
    :param label:
        input label.
    :param K:
        number of shapelets that finally learned.
    :param C:
        number of shapelet candidates in learning procedure.
    :param num_segment:
    :param seg_length:
    :param data_size:
    :param p:
    :param lr:
    :param alpha:
    :param beta:
    :param num_batch:
    :param gpu_enable:
    :param measurement:
    :param kwargs:
    :return:
    """
    cands = generate_shapelet_candidate(time_series_set=time_series_set, num_segment=num_segment,
                                        seg_length=seg_length, candidate_size=C, **kwargs)
    parmap = ParMap(
        work=__shapelet_candidate_loss_factory(
            time_series_set=time_series_set, label=label, num_segment=num_segment, seg_length=seg_length,
            data_size=data_size, p=p, lr=lr, alpha=alpha, beta=beta, num_batch=num_batch,
            gpu_enable=gpu_enable, measurement=measurement, **kwargs
        ),
        monitor=parallel_monitor(msg='learning time-aware shapelets', size=len(cands),
                                 debug=kwargs.get('debug', True)),
        njobs=kwargs.get('njobs', NJOBS)
    )
    result = sorted(parmap.run(data=cands), key=lambda x: x[3])
    ret = []
    for (cand, local_factor, global_factor, loss, main_loss, penalty) in result:
        ret.append((cand, local_factor, global_factor, loss))
    return sorted(ret, key=lambda x: x[-1])[:K]
