import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

import math


def get_positional_encoding(number_of_seq, hidden_dim=512, device='cpu'):
    '''
        * math vs numpy
            -> math can calc 'scalar' only, but numpy can calc not only 'scalar', but also 'matrix'
    '''
    # 1. python
    def calc_angle(pos, h_dim):
        return [pos / math.pow(10000, (2*(i//2))/h_dim) for i in range(h_dim)]
    def make_seq_list(n_seq, h_dim):
        return np.array([calc_angle(pos, h_dim) for pos in range(n_seq)]) 

    sinusoidal_tbl = make_seq_list(number_of_seq, hidden_dim)
    sinusoidal_tbl[:, 0::2] = np.sin(sinusoidal_tbl[:, 0::2])
    sinusoidal_tbl[:, 1::2] = np.cos(sinusoidal_tbl[:, 1::2])

    sinusoidal_tbl = torch.from_numpy(sinusoidal_tbl)
    sinusoidal_tbl = sinusoidal_tbl.type(torch.float32)
    sinusoidal_tbl = sinusoidal_tbl.to(device)

    # 2. C
    if 0:
        sinusoidal_tbl = torch.zeros((number_of_seq, hidden_dim), device=device)
        for pos in range(number_of_seq):
            for i in range(hidden_dim):
                if i % 2 == 0:
                    sinusoidal_tbl[pos, i] = math.sin(pos / math.pow(10000, (2*(i//2))/hidden_dim))
                else:
                    sinusoidal_tbl[pos, i] = math.cos(pos / math.pow(10000, (2*(i//2))/hidden_dim))

    # for debugging
    if 0:
        print(sinusoidal_tbl)
        plt.pcolormesh(sinusoidal_tbl, cmap='RdBu')
        plt.xlabel('Depth')
        plt.xlim((0, hidden_dim))
        plt.ylabel('Position')
        plt.colorbar()
        plt.show()

    return sinusoidal_tbl