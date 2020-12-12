import warnings
warnings.simplefilter('ignore')

import torch
import numpy as np
from torch import nn
from torch.nn import init
import pandas as pd
from torch.utils import data
from torchvision.transforms import Compose, ToTensor


def OneHotPandasString(seqs, mapp):
    for k, v in mapp.items():
        seqs = seqs.str.replace(k, str(v))
    seqs = seqs.apply(lambda x: np.array(list(x)).astype(int))
    seqs = np.stack(seqs.values)
    return seqs

def OneHotEncode(seqs, mapp):
    encs = []
    for seq in seqs.values:
        encs.append(np.array([int(mapp[s]) for s in seq]))
    encs = np.stack(encs)
    return encs

class Pad:
    def __init__(self, max_len, pad_symbol='X'):
        self.max_len = max_len
        self.Pad_symbol = pad_symbol

    def __call__(self, seqs):
        return seqs.apply(lambda x: x + self.Pad_symbol * (self.max_len - len(x)))


class PadNumpyRight:
    def __init__(self, max_len, pad_value=0):
        self.max_len = max_len
        self.Pad_value = pad_value

    def __call__(self, seqs):
        N, D, L = seqs.shape
        z = np.zeros([N, D, self.max_len-L])
        return np.concatenate([seqs, z], axis=-1)


class Memory_container(dict):
    def __init__(self, memory_attrs):
        for ma in memory_attrs:
            self.update({ma: None})

class Mask():
    def __init__(self, max_num_islands=5, max_island_len=5, p=0.8, mask_symb=4):
        self.num_islands = np.arange(1,max_num_islands)
        self.island_len = np.arange(1,max_island_len)
        self.prob = p
        self.mask_symb = mask_symb

    def __call__(self, seqs):
        L = seqs.shape[0]
        if np.random.uniform() < self.prob:
            isl_num = np.random.choice(self.num_islands)
            isl_sizes = np.random.choice(self.island_len, size=isl_num)
            positions = np.random.choice(L-30, size=isl_num)
            mask_idx = np.concatenate([np.arange(pos, pos+size) for pos, size in zip(positions, isl_sizes)])
            seqs[mask_idx] = self.mask_symb
        return seqs


class Roll():
    def __init__(self, max_roll=20, p=0.3, memory_container={'rolls': None}, from_memory=False):
        assert 'rolls' in memory_container.keys()
        self.max_roll = max_roll
        self.prob = p
        self.memory = memory_container
        self.from_memory = from_memory

    def __call__(self, seqs):
        if self.from_memory:
            rolls = self.memory['rolls']
        else:
            if np.random.uniform() < self.prob:
                rolls = int(np.random.beta(a=2, b=3) * 20)
            else:
                rolls = 0
            self.memory['rolls'] = rolls
            self.memory['prescribed'] = None
        # print('Seqs shape:', seqs.shape)
        seqs = torch.roll(seqs, shifts=rolls, dims=0 if len(seqs.shape) < 2 else 1)
        return seqs

class Prescribe():
    def __init__(self, p, memory_container):
        assert 'rolls' in memory_container.keys()
        self.prob = p
        self.memory = memory_container

    def __call__(self, seqs):
        rolls = self.memory['rolls']
        if rolls > 3:
            if np.random.uniform() < self.prob:
                prefix_len = int(np.random.beta(a=2, b=3) * rolls)
                prefix = np.random.choice([0, 1, 2, 3], size=prefix_len, p=[0.45, 0.25, 0.15, 0.15])
                seqs[..., (rolls - prefix_len):rolls] = torch.Tensor(prefix)
                # print(rolls, prefix_len)
                self.memory['prescribed'] = prefix_len
        return seqs


class PrescribeShadow():
    def __init__(self, insert_element, memory_container):
        assert 'rolls' in memory_container.keys()
        assert 'prescribed' in memory_container.keys()
        self.elem = insert_element
        self.memory = memory_container

    def __call__(self, seqs):
        added = self.memory['prescribed']
        rolls = self.memory['rolls']
        # print('#####')
        # print(rolls, added)
        if added:
            if added > 0:
                seqs[..., (rolls - added):rolls] = self.elem
        return seqs

class scatter_torch():
    def __init__(self, dims):
        self.dims = dims

    def __call__(self, idx):
        o1 = torch.zeros(*idx.shape, self.dims).type_as(idx)
        return o1.scatter_(-1, idx.unsqueeze(-1), 1)

def to_tensor(t):
    if type(t) == int:
        return torch.LongTensor([t])
    elif type(t) == float:
        return torch.Tensor([t])
    elif 'int' in str(t.dtype):
        return torch.LongTensor(t)
    return torch.Tensor(t)

def crop_out_padding(t):
    return t[..., :-1]

def to_float(t):
    return t.float()


def Upper(seqs):
    return seqs.apply(lambda x: x.upper())


def to_cuda(func):
    def wrap(*args, **kwargs):
        return func(*args, **kwargs).cuda()

    return wrap

def weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
        init.kaiming_normal_(m.weight)
