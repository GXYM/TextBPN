# TextBPN
Adaptive Boundary Proposal Network for Arbitrary Shape Text Detection； Accepted by ICCV2021.  
![](https://github.com/GXYM/TextBPN/blob/main/vis/1.png)  

The codes and paper of TextBPN++ has been released at:[TextBPN-Puls-Plus](https://github.com/GXYM/TextBPN-Puls-Plus)

## 1.Prerequisites 
  python 3.9;  
  PyTorch 1.7.0;   
  Numpy >=1.2.0   
  CUDA 11.1;  
  GCC >=10.0;  
  *opencv-python < 4.5.0*  
  NVIDIA GPU(with 11G or larger GPU memory for inference);  

## 2.Dataset Links  
1. [CTW1500](https://drive.google.com/file/d/1A2s3FonXq4dHhD64A2NCWc8NQWMH2NFR/view?usp=sharing)   
2. [TD500](https://drive.google.com/file/d/1ByluLnyd8-Ltjo9AC-1m7omZnI-FA1u0/view?usp=sharing)  
3. [Total-Text](https://drive.google.com/file/d/17_7T_-2Bu3KSSg2OkXeCxj97TBsjvueC/view?usp=sharing) 

## 3.Models
 *  [Total-Text model](https://drive.google.com/file/d/13ZrWfpv2dJqdv0cDf-hE0n5JQegNY0-i/view?usp=sharing) (pretrained on ICDAR2017-MLT)
 *  [CTW-1500 model](https://drive.google.com/file/d/13ZrWfpv2dJqdv0cDf-hE0n5JQegNY0-i/view?usp=sharing) (pretrained on ICDAR2017-MLT)
 *  [MSRA-TD500 model](https://drive.google.com/file/d/13ZrWfpv2dJqdv0cDf-hE0n5JQegNY0-i/view?usp=sharing) (pretrained on ICDAR2017-MLT)  

## 4.Running Evaluation
run:  
```
sh eval.sh
```
The details are as follows:  
```
#!/bin/bash
##################### Total-Text ###################################
# test_size=[640,1024]--cfglib/option
CUDA_LAUNCH_BLOCKING=1 python eval_textBPN.py --exp_name Totaltext --checkepoch 390 --dis_threshold 0.3 --cls_threshold 0.825 --test_size 640 1024 --gpu 1

###################### CTW-1500 ####################################
# test_size=[640,1024]--cfglib/option
# CUDA_LAUNCH_BLOCKING=1 python eval_textBPN.py --exp_name Ctw1500 --checkepoch 560 --dis_threshold 0.3 --cls_threshold 0.8 --test_size 640 1024 --gpu 1

#################### MSRA-TD500 ######################################
# test_size=[640,1024]--cfglib/option
#CUDA_LAUNCH_BLOCKING=1 python eval_textBPN.py --exp_name TD500 --checkepoch 680 --dis_threshold 0.3 --cls_threshold 0.925 --test_size 640 1024 --gpu 1

```  
## 5.Experiments results
![](https://github.com/GXYM/TextBPN/blob/main/vis/2.png)
![](https://github.com/GXYM/TextBPN/blob/main/vis/3.png)


