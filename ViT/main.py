import torch
import torch.nn as nn
import torch.optim as optim

import pytorch_model_summary

import os
import sys

from train_eval.train_eval import train_vit, eval_vit

root_dir = os.path.dirname(os.path.realpath("main.py"))
sys.path.insert(0, root_dir)

from dataset.cifar10 import get_dataloader_cifar10
from utils.utils import init_weights, get_official_pretrained_vit_models
from utils.arg_parser import get_vit_args, print_vit_args
from train_eval.train_eval import train_vit, eval_vit

from vision_transformer import VisionTransformer


# get hyper-parameter of vit
args = get_vit_args()


if __name__ == "__main__":

    # for reproducibility
    torch.manual_seed(777)
    if args.gpu_mode == False:
        device = 'cpu'
    else:
        device = 'cuda' if torch.cuda.is_available() == True else 'cpu'
        if device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.manual_seed_all(777)

    # data loader
    trainloader = get_dataloader_cifar10(opt_data="train", batch_size=args.batch_size)
    testloader = get_dataloader_cifar10(opt_data="test", batch_size=args.batch_size)

    # In order to compare reproduce model with offical
    if args.official_name != '':
        model = get_official_pretrained_vit_models(args.official_name, args.n_cls, args.img_size)
    else:
        model = VisionTransformer(args.n_cls,
                                args.n_heads,
                                args.n_layers,
                                args.h_d,
                                (args.h_d*args.mlp_ratio),
                                args.img_size,
                                args.img_size,
                                patch_size=args.patch_size,
                                dropout_rto=args.d_o,
                                device=device)

    # init model weights
    model.apply(init_weights)

    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), 
                                lr=args.lr, 
                                betas=(0.9, 0.999),
                                weight_decay=args.w_d)
    else: # sgd
        optimizer = optim.SGD(model.parameters(), 
                                lr=args.lr, 
                                momentum=args.momentum)

    # lr scheduler
    if args.lr_scheduler == 'cosine':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=(args.epoch//5),
                                                            eta_min=(args.lr*1e-2))
    else: # linear (will be added..)
        pass

    # loss function
    loss_ft = nn.CrossEntropyLoss()

    # check '#' of model parameters
    if args.debug_mode:
        print_vit_args(args) # print all parameters
        print(pytorch_model_summary.summary(model, 
                                            torch.zeros(args.batch_size, 3, args.img_size, args.img_size, device=device), 
                                            max_depth=None,
                                            show_parent_layers=False,#True,
                                            show_input=True))
    else:
        print_vit_args(args) # print all parameters
        # train vit
        train_vit(model, 
                optimizer, 
                lr_scheduler,
                loss_ft, 
                trainloader, 
                args.n_cls,
                args.epoch, 
                args.batch_size, 
                device=device)
        
        # evaluate vit
        eval_vit(model,
                testloader,
                device=device)