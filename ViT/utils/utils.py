import torch
import torch.nn as nn


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