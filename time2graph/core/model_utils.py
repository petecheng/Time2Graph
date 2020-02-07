# -*- coding: utf-8 -*-
from torch.utils.data import Dataset
from torch.utils.data.sampler import WeightedRandomSampler


class NumpyDataset(Dataset):
    """ Dataset wrapping numpy ndarrays
    Each sample will be retrieved by indexing numpy-arrays along the first dimension.

    Arguments:
        *ndarrays (numpy-ndarray): ndarrays that have the same size of the first dimension.
    """
    def __init__(self, *ndarrays):
        assert all(ndarrays[0].shape[0] == ndarray.shape[0] for ndarray in ndarrays)
        self.ndarrays = ndarrays

    def __getitem__(self, idx):
        return tuple(ndarray[idx] for ndarray in self.ndarrays)

    def __len__(self):
        return self.ndarrays[0].shape[0]


class StratifiedSampler(WeightedRandomSampler):
    """
    Stratified Sampler in torch.
    """
    def __init__(self, label, num_class):
        self.num_class = num_class
        weights = self.__get_weight(label=label)
        super(StratifiedSampler, self).__init__(weights=weights, num_samples=len(weights))

    def __get_weight(self, label):
        num_class = self.num_class
        cnt = [0] * num_class
        for lb in label:
            cnt[lb] += 1
        weight_per_class, total = [0.0] * num_class, float(sum(cnt))
        for k in range(num_class):
            weight_per_class[k] = total / float(cnt[k])
        ret = [0.0] * len(label)
        for idx, val in enumerate(label):
            ret[idx] = weight_per_class[val]
        return ret
