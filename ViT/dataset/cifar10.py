import os
import sys
import platform

import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

if platform.system() == 'Windows':
    ROOT_DIR = "D:/.data"
else:
    ROOT_DIR = "/home/jw/vscode/dataset/cifar10"


cls_cifar10 = ('plane', 
'car', 
'bird', 
'cat', 
'deer', 
'dog', 
'frog', 
'horse', 
'ship', 
'truck')

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


def get_dataloader_cifar10(opt_data="train", batch_size=64, device='cpu'):
    if opt_data == "train":
        trainset_cifar10 = torchvision.datasets.CIFAR10(root=ROOT_DIR,
        train=True, download=True, transform=transform)
        return DataLoader(trainset_cifar10, batch_size=batch_size, shuffle=True)
    else: #"test"
        testset_cifar10 = torchvision.datasets.CIFAR10(root=ROOT_DIR,
        train=False, download=True, transform=transform)
        return DataLoader(testset_cifar10, batch_size=batch_size, shuffle=False)