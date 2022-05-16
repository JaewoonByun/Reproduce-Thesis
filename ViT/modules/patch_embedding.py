import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt


class PatchEmbedding(nn.Module):
    def __init__(self, width, height, patch_size, hidden_dim, in_chans=3, device='cpu'):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.number_of_patches = int((width*height) / (patch_size*patch_size))
        
        self.split_patch = nn.Conv2d(in_channels=in_chans, 
                                    out_channels=hidden_dim, 
                                    kernel_size=patch_size, 
                                    stride=patch_size,
                                    device=device)

    def forward(self, x):
        # 1. split patches using convolution
        x = self.split_patch(x)
        #print(x.shape)
        
        # 2. Linear projection of flatten patches
        x = torch.flatten(x, start_dim=2)
        #print(x.shape)

        x = torch.transpose(x, 2, 1)
        #print(x.shape)
        
        return x
