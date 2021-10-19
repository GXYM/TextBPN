###################################################################
# File Name: gcn.py
# Author: Zhongdao Wang
# mail: wcd17@mails.tsinghua.edu.cn
# Created Time: Fri 07 Sep 2018 01:16:31 PM CST
###################################################################

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class Rnn(nn.Module):
    def __init__(self, input, state_dim):
        super(Rnn, self).__init__()
        self.bn0 = nn.BatchNorm1d(input, affine=False)
        self.rnn = nn.LSTM(input, state_dim, 1, dropout=0.1, bidirectional=True)
        self.prediction = nn.Sequential(
            nn.Conv1d(state_dim*2, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 2, 1))

    def forward(self, x, adj):
        x = self.bn0(x)
        x = x.permute(2, 0, 1)
        x, _ = self.rnn(x)
        x = x.permute(1, 2, 0)
        pred = self.prediction(x)

        return pred
