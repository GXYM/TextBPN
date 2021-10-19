#!/bin/bash
cd ../
CUDA_LAUNCH_BLOCKING=1 python train_textBPN.py --exp_name Synthtext --net resnet50 --max_epoch 2 --batch_size 16 --lr 0.001 --gpu 0 --input_size 512 --save_freq 1 --num_workers 24 #--viz --viz_freq 10000
#--resume model/Synthtext/TextBPN_resnet50_0.pth --start_epoch 1
