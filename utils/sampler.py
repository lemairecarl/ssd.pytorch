import torch
from torch.utils.data.sampler import Sampler


class RandomSampler(Sampler):
    r"""Samples elements randomly, without replacement.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        cpu = torch.device('cpu')
        return iter( torch.randperm( len(self.data_source), device = cpu).tolist())

    def __len__(self):
        return len(self.data_source)
