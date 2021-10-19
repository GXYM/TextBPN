#!/bin/bash
##################### Total-Text ###################################
# test_size=(640,1024)--cfglib/option
CUDA_LAUNCH_BLOCKING=1 python eval_textBPN.py --exp_name Totaltext --checkepoch 390 --dis_threshold 0.3 --cls_threshold 0.825 --test_size 640 1024 --gpu 1

###################### CTW-1500 ####################################
# test_size=(640,1024)--cfglib/option
# CUDA_LAUNCH_BLOCKING=1 python eval_textBPN.py --exp_name Ctw1500 --checkepoch 560 --dis_threshold 0.3 --cls_threshold 0.8 --test_size 640 1024 --gpu 1

#################### MSRA-TD500 ######################################
# test_size=(640,1024)--cfglib/option
#CUDA_LAUNCH_BLOCKING=1 python eval_textBPN.py --exp_name TD500 --checkepoch 680 --dis_threshold 0.3 --cls_threshold 0.925 --test_size 640 1024 --gpu 1
