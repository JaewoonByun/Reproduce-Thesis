import torch
import torch.nn as nn
'''
    for debugging # of parameters about 'ViT-L16'
    in thesis,
        # of ViT-B16: 86M
            -> layer:12 |   hidden size(D):768  |   MLP size:3072   |   head:12
        # of 'ViT-L16': 30'7'M
            -> layer:24 |   hidden size(D):1024  |   MLP size:4096   |   head:16
        # of ViT-H16: 632M
            -> layer:32 |   hidden size(D):1280  |   MLP size:5120   |   head:16

    but, in my ViT models
        # of ViT-B16: 86M
            -> layer:12 |   hidden size(D):768  |   MLP size:3072   |   head:12
        # of 'ViT-L16': 30'4'M
            -> layer:24 |   hidden size(D):1024  |   MLP size:4096   |   head:16
        # of ViT-H16: 632M
            -> layer:32 |   hidden size(D):1280  |   MLP size:5120   |   head:16
'''
from pytorch_pretrained_vit import ViT


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.constant_(m.bias, 0.01) #m.bias.data.fill_(0.01)
    if type(m) == nn.LayerNorm:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    if type(m) == nn.Conv2d:
        nn.init.uniform_(m.weight)
        

def get_one_hot_encoding(label, n_classes=10, batch_size=64, device='cpu'):
    # one-hot encoding
    label = label.unsqueeze(dim=1)
    one_hot = torch.zeros((batch_size, n_classes)).to(device)
    one_hot.scatter_(1, label, 1) # src: can set constant value !
    #print(one_hot)
    return one_hot


'''
    pre-trained vit models(name):
        B_16, B_32, L_16, L_32, B_16_imagenet1k, B_32_imagenet1k, L_16_imagenet1k, L_32_imagenet1k
'''
def get_official_pretrained_vit_models(name, n_classes=1000, img_size=384):
    pretrained_vit = ViT(name=name, num_classes=n_classes, image_size=img_size, pretrained=True)
    return pretrained_vit