
import torch
import numpy as np
from torch import nn
from torch.utils import data
from utils import *


class DoubleMotifs(data.Dataset):
    def __init__(self, df, positive_rate=0.2, preprocess_x=None, preprocess_y=None, transforms=None, transforms_x2=None,
                 transforms_y=None):
        self.data = df
        self.idx = np.arange(df.shape[0])
        self.X1 = preprocess_x(df['HTH_seq']) if preprocess_x else df['HTH_seq'].values
        self.X2 = preprocess_y(df['aligned_motifs']) if preprocess_y else df['aligned_motifs'].values
        self.Y = preprocess_y(df['sites']) if preprocess_y else df['site'].values

        self.X_len = df['HTH_seq'].apply(len)
        self.Y_len = df['sites'].apply(len)
        self.Positives = positive_rate
        self.transforms = transforms
        self.transforms_x2 = transforms_x2
        self.transforms_y = transforms_y

    def __getitem__(self, idx):
        x, s = self.X1[idx], self.Y[idx]
        if np.random.uniform() < self.Positives:
            x2 = self.X2[idx]
            y = 1.
        else:
            mask = ~np.all(self.Y == s, axis=1)
            idx2 = np.random.choice(mask.sum())
            x2 = self.X2[mask][idx2]
            y = 0.
        if self.transforms:
            x = self.transforms(x)
        if self.transforms_x2:
            x2 = self.transforms_x2(x2)
        if self.transforms_x2:
            s = self.transforms_x2(s)
        if self.transforms_y:
            y = self.transforms_y(y)
        return x, x2, y, s, idx

    def __len__(self):
        return len(self.idx)


