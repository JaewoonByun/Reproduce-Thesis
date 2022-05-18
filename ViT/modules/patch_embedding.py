import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt


class PatchEmbedding(nn.Module):
    def __init__(self, width, height, patch_size, hidden_dim, split_mode='conv2d', in_chans=3, device='cpu'):
        super(PatchEmbedding, self).__init__()
        self.split_mode = split_mode
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.n_patches = int((width*height) / (patch_size*patch_size))
        self.split_patch = nn.Conv2d(in_channels=in_chans, 
                            out_channels=hidden_dim,
                            kernel_size=patch_size,
                            stride=patch_size,
                            device=device)
        self.linear_proj = nn.Linear(self.in_chans*(self.patch_size**2), hidden_dim)

    def forward(self, x):
        # 1. split patches using convolution
        # use nn.Linear
        if self.split_mode == 'conv2d': # nn.Conv2d
            # 1. Split patch
            x = self.split_patch(x)
            #print(x.shape)

            # 2. Linear projection of flatten patches
            x = torch.flatten(x, start_dim=2)
            x = torch.transpose(x, 2, 1)
            #print(x.shape)
        else: # nn.Linear
            # 1. Split patch
            x = x.view(-1, self.in_chans, self.n_patches, (self.patch_size)**2)
            x = x.permute(0, 2, 1, 3).contiguous()
            #print(x.shape)

            # 2. Linear projection of flatten patches
            x = torch.flatten(x, start_dim=2)
            x = self.linear_proj(x)
            #print(x.shape)
        
        return x
