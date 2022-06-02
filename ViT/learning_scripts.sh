#!/bin/sh

python main.py --epoch 10 --batch_size 10 --lr 3e-4 --optim sgd --w_d 0.1 --d_o 0.1 --h_d 512 --n_layers 8 --n_heads 8 --gpu_mode False