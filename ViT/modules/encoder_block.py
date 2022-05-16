import torch
import torch.nn as nn

import os
import sys

root_dir = os.path.dirname(os.path.realpath("main.py"))
sys.path.insert(0, root_dir)

from modules.attention import MultiHeadAttentionLayer
from modules.mlp import MLP


class Encoder(nn.Module):
    def __init__(self, 
                 n_heads, 
                 hidden_dim, 
                 mlp_hidden_dim,
                 dropout_rto=0.5, 
                 device='cpu'):
        super(Encoder, self).__init__()
        # encoder
        # -> layer_norm have to divide for each sub-layers in encoder !
        self.ln_msa = nn.LayerNorm(hidden_dim, device=device)
        self.ln_mlp = nn.LayerNorm(hidden_dim, device=device)
        self.msa = MultiHeadAttentionLayer(
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            dropout_ratio=dropout_rto,
            device=device
        )
        self.mlp = MLP(hidden_dim, mlp_hidden_dim, hidden_dim, dropout_rto=dropout_rto, device=device)
        
    def forward(self, x):
        # sub-layer 1
        if 0: # prev.
            q, k, v = [self.ln_msa(x) for _ in range(3)]
            msa, _ = self.msa(q, k, v)
        else:
            x = self.ln_msa(x)
            msa, _ = self.msa(x, x, x) # q, k, v == 'x'
        x = x + msa
                
        # sub-layer 2
        x = x + self.mlp(self.ln_mlp(x))
        return x


class EncoderBlock(nn.Module):
    def __init__(self,
                 n_heads, 
                 n_layers, 
                 hidden_dim, 
                 mlp_hidden_dim,
                 dropout_rto=0.5, 
                 device='cpu'):
        super(EncoderBlock, self).__init__()
        self.n_layers = n_layers
        self.enc_blks = nn.ModuleList(
            [Encoder(n_heads,
                    hidden_dim,
                    mlp_hidden_dim,
                    dropout_rto=dropout_rto,
                    device=device
                    ) for _ in range(self.n_layers)])

    def forward(self, x):
        for enc in self.enc_blks:
            x = enc(x)
        return x