#!/bin/bash
cd ../
CUDA_LAUNCH_BLOCKING=1 python train_textBPN.py --exp_name Totaltext --net resnet50 --max_epoch 660 --batch_size 12 --gpu 0 --input_size 640 --optim Adam --lr 0.001 --num_workers 24  
#--resume model/MLT2017/TextBPN_resnet50_100.pth 
#--start_epoch 300
#--viz 
