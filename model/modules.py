import os
import sys
import numpy as np

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
import torch.distributed as dist

class FlattenFC(nn.Module):
    def __init__(self, d_in, d_out, keep_dims=False):
        super(FlattenFC, self).__init__()
        self.keep_dims = keep_dims
        self.linear = nn.Linear(d_in, d_out)
    
    def forward(self, input):
        
        dims = input.size()
        
        input = input.view(dims[0], -1)
        
        input = self.linear(input)
        
        if self.keep_dims:
            input = input.view(dims)
        
        return input

class Transpose(nn.Module):
    def __init__(self, transpose):
        super(Transpose, self).__init__()
        self.transpose = transpose

    def forward(self, input):
        return input.transpose(*self.transpose)

class CausalConv1d(nn.Module):
    def __init__(self, d_in, d_out, kernel_size=1, stride=1, dilation=1, padding=0):
        super(CausalConv1d, self).__init__()
        
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels = d_in, out_channels = d_out, kernel_size = kernel_size, stride = stride, dilation = dilation)
    
    def forward(self, input):
        # only left-padding instead of padding to both sides because of masking
        input = F.pad(input, (self.pad,0))
        
        out = self.conv(input)
        
        return out
