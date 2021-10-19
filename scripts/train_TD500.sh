#!/bin/bash
cd ../
CUDA_LAUNCH_BLOCKING=1 python train_textBPN.py --exp_name TD500 --net resnet50 --max_epoch 2000 --batch_size 12 --gpu 0 --input_size 640 --optim Adam --lr 0.001 --num_workers 28  --resume model/MLT2017/TextBPN_resnet50_100.pth --save_freq 20
