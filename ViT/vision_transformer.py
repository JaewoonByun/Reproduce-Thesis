import torch
import torch.nn as nn

import os
import sys

root_dir = os.path.dirname(os.path.realpath("main.py"))
sys.path.insert(0, root_dir)

from modules.patch_embedding import PatchEmbedding
from modules.encoder_block import EncoderBlock
from modules.positional_encoding import get_positional_encoding


class VisionTransformer(nn.Module):
    def __init__(self,
                 n_classes,
                 n_heads,
                 n_layers,
                 hidden_dim,
                 mlp_hidden_dim,
                 img_width,
                 img_height,
                 in_chans=3,
                 patch_size=16,
                 dropout_rto=0.5,
                 device='cpu'):
        super(VisionTransformer, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.n_patches = (img_width // patch_size)**2
        self.patches = PatchEmbedding(img_width, img_height, patch_size, hidden_dim, in_chans=in_chans, device=device)
        self.posEnc = get_positional_encoding(self.n_patches+1, hidden_dim=hidden_dim, device=device)
        self.cls = nn.Parameter(torch.zeros(1, hidden_dim))
        self.encoderBlk = EncoderBlock(
            n_heads,
            n_layers,
            hidden_dim,
            mlp_hidden_dim,
            dropout_rto=dropout_rto,
            device=device
        )

        self.layer_norm = nn.LayerNorm(hidden_dim, device=device)
        self.mlp_head = nn.Linear(hidden_dim, n_classes, device=device)
            
    def forward(self, x):
        # 1. patch embedding
        cls = self.cls.expand((x.size(0), 1, self.hidden_dim)).to(self.device)
        x = torch.cat((cls, self.patches(x)), dim=1)
        x = x + self.posEnc
        #print('patch embedding:{0}'.format(x.shape))
        
        # 2. transformer encoder block
        x = self.encoderBlk(x)
        #print('encoder blk:{0}'.format(x.shape))
        
        # 3. fc-layers to match number of classes
        cls_embed_token = x[:, 0]
        #print('embedded patch of class: {0}'.format(cls_embed_token))
        result = self.mlp_head(self.layer_norm(cls_embed_token))
        #print('mlp head(final):{0}'.format(result.shape))

        return result
        