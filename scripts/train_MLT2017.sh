#!/bin/bash
cd ../
CUDA_LAUNCH_BLOCKING=1 python train_textBPN.py --exp_name MLT2017 --net resnet50 --max_epoch 200 --batch_size 12 --gpu 0 --input_size 640 --optim Adam --lr 0.001 --num_workers 24 --resume model/MLT2017/TextBPN_resnet50_100.pth
#--viz --viz_freq 400
