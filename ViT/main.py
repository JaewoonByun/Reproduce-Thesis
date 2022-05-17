import torch
import torch.nn as nn
import torch.optim as optim

import pytorch_model_summary

import os
import sys

from train_eval.train_eval import train_vit, eval_vit

root_dir = os.path.dirname(os.path.realpath("main.py"))
sys.path.insert(0, root_dir)

from dataset.cifar10 import get_dataloader_cifar10, cls_cifar10
from utils.utils import init_weights, get_official_pretrained_vit_models
from train_eval.train_eval import train_vit, eval_vit
from vision_transformer import VisionTransformer


# hyper-parameters
DEBUG_MODE = False#True
USE_OFFICIAL_MODEL = False#True
offical_vit_name = "B_16_imagenet1k"

epoch = 20
batch_size = 100
learning_rate = 2e-4 # 2*10^-4
weight_decay = 0.1
momentum = 0.9

img_width = 32#384
img_height = 32#384
img_patch_size = 16

N_CLASSES = len(cls_cifar10)
HIDDEN_DIM = 768
MLP_HIDDEN_DIM = HIDDEN_DIM * 4
ENC_LAYERS = 12
ENC_HEADS = 12
ENC_DROPOUT = 0.1


if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() == True else 'cpu'
    # for reproducibility
    torch.manual_seed(777)
    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.manual_seed_all(777)

    trainloader = get_dataloader_cifar10(opt_data="train", batch_size=batch_size)
    testloader = get_dataloader_cifar10(opt_data="test", batch_size=batch_size)

    # In order to compare '#' of parameters with offical ViT models
    if USE_OFFICIAL_MODEL:
        model = get_official_pretrained_vit_models(offical_vit_name, N_CLASSES, img_width)
    else: # 
        model = VisionTransformer(N_CLASSES,
                                ENC_HEADS,
                                ENC_LAYERS,
                                HIDDEN_DIM,
                                MLP_HIDDEN_DIM,
                                img_width,
                                img_height,
                                patch_size=img_patch_size,
                                dropout_rto=ENC_DROPOUT,
                                device=device)

    # init model weights
    model.apply(init_weights)

    optimizer_adam = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer_sgd = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    loss_ft = nn.CrossEntropyLoss()

    # for verify vit parameters
    if DEBUG_MODE:
        print(pytorch_model_summary.summary(model, 
                                            torch.zeros(batch_size, 3, img_width, img_height, device=device), 
                                            max_depth=None,
                                            show_parent_layers=False,#True,
                                            show_input=True))
    else:
        # train vit
        train_vit(model, 
                optimizer_adam, 
                loss_ft, 
                trainloader, 
                N_CLASSES,
                epoch, 
                batch_size, 
                device=device)
        
        # evaluate vit
        eval_vit(model,
                testloader,
                device=device)