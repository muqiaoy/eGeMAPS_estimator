#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2022 Muqiao Yang <muqiaoy@andrew.cmu.edu>

import numpy as np
import torch
import torchaudio
import torch.nn as nn
from torchaudio.pipelines import Wav2Vec2Bundle
import opensmile
from .Demucs.utils import capture_init

from . import modules as custom_nn
import collections
from torch.nn import functional as F
import torch.distributed as dist

def output_to_dist(output, dim=-1):
    z_size = output.size(dim)//2
    mean, log_var = torch.split(output, z_size, dim=dim)
    return torch.distributions.Normal(mean, torch.exp(0.5*log_var))

class VAE(nn.Module):
    @capture_init
    def __init__(self, z_size, local_z_size):
        super(VAE, self).__init__()
        self.encoder = Encoder(z_size, local_z_size)
        self.decoder = Decoder(local_z_size)
        self.predictor = Predictor(z_size, local_z_size)
        
        self.VAEOutput = collections.namedtuple("VAEOutput", ["encoder_out", "decoder_out", "predictor_out"])
    
    def forward(self, input, annealing=0):
        
        # print(input.shape)
        encoder_out = self.encoder(input)
        # print(encoder_out.local_dist)
        # print(encoder_out.local_sample.shape)
        # print(encoder_out.global_dist)
        # print(encoder_out.global_sample.shape)
        predictor_out = self.predictor(input, encoder_out.local_sample)
        # print(predictor_out)
        
        decoder_out = self.decoder(encoder_out.local_sample)
        # print(decoder_out.shape)
        # assert False
        assert torch.min(decoder_out) >= 0.
        assert torch.max(decoder_out) <= 1.
        
        return self.VAEOutput(encoder_out, decoder_out, predictor_out)
    
class Encoder(nn.Module):
    def __init__(self, z_size, local_z_size):
        super(Encoder, self).__init__()
        
        self.EncoderOutput = collections.namedtuple("EncoderOutput", ["local_dist", "local_sample", "global_dist", "global_sample"])
        
        
        
        self.global_net = nn.Sequential(
            custom_nn.Transpose((1, 2)),
            nn.Conv1d(201, z_size//4, kernel_size = 3, stride = 1),
            nn.Tanh(),
            
            nn.BatchNorm1d(z_size//4),
            
            nn.Conv1d(z_size//4, z_size//2, kernel_size = 3, stride = 1),
            nn.Tanh(),
            nn.BatchNorm1d(z_size//2),
            
            nn.Conv1d(z_size//2, z_size, kernel_size = 3, stride = 1),
            nn.Tanh(),
            nn.BatchNorm1d(z_size),
            
            nn.Conv1d(z_size, 2*z_size, kernel_size = 1, stride = 1),
            custom_nn.Transpose((1, 2)),
            
        )
        
        self.local_net = nn.Sequential(
            custom_nn.Transpose((1, 2)),
            nn.Conv1d(201+z_size, local_z_size, kernel_size = 1, stride = 1),
            nn.Tanh(),
            nn.BatchNorm1d(local_z_size),
            
            nn.Conv1d(local_z_size, local_z_size, kernel_size = 1, stride = 1),
            nn.Tanh(),
            nn.BatchNorm1d(local_z_size),
            
            nn.Conv1d(local_z_size, 2*local_z_size, kernel_size = 1, stride = 1),
            custom_nn.Transpose((1, 2)),
        )
        self.light_dropout = nn.Dropout(0.3)
        
        self.Sigmoid = nn.Sigmoid()

    def forward(self, input):
        
        # input is a tensor of batch x time x features
        assert len(input.size()) == 3
        
        global_out = self.global_net(input)
        # global average pooling
        global_out = torch.mean(global_out, dim=1)
        
        # for testng purposes of speaker transformation
        # ujson.dump(global_out.tolist(), open("experiments/global_out.json", 'w'))
        # global_out = torch.FloatTensor(ujson.load(open('experiments/global_out.json', 'r')))
        # global_out = global_out + torch.rand(global_out.size())
        
        global_out = global_out.unsqueeze(1)
        global_out = global_out.repeat(1,input.size(1),1)
        
        global_dist = output_to_dist(global_out)
        
        global_z_sample = global_dist.rsample()
        
        resized_sample = global_z_sample
        
        if self.training:
            local_out = self.local_net(torch.cat((F.dropout(input, p=0.2, training=True), resized_sample), 2))
        else:
            local_out = self.local_net(torch.cat((F.dropout(input, p=0.2, training=True), resized_sample), 2))
        
        local_dist = output_to_dist(local_out)
        
        local_z_sample = local_dist.rsample()
        return self.EncoderOutput(local_dist=local_dist, local_sample=local_z_sample, global_dist=global_dist, global_sample=global_z_sample)

class Decoder(nn.Module):
    def __init__(self, local_z_size):
        super(Decoder, self).__init__()
        
        self.fc = nn.Sequential(
            custom_nn.Transpose((1,2)),
            nn.Conv1d(local_z_size, 201, kernel_size = 1, stride = 1),
            nn.Tanh(),
            nn.BatchNorm1d(201),
            
            nn.Conv1d(201, 201, kernel_size = 1, stride = 1),
            nn.Tanh(),
            nn.BatchNorm1d(201),
            
            nn.Conv1d(201, 201, kernel_size = 1, stride = 1),
            nn.Tanh(),
            nn.BatchNorm1d(201),
            
            nn.Conv1d(201, 201, kernel_size=1, stride=1),
            nn.Sigmoid(),
            custom_nn.Transpose((1,2)),
        )
    
    def forward(self, input):        
        out = self.fc(input)
        
        return out

class Predictor(nn.Module):
    def __init__(self, z_size, local_z_size):
        super(Predictor, self).__init__()
        
        self.z_recon_fc = nn.Sequential(
            custom_nn.Transpose((1,2)),
            nn.Conv1d(local_z_size, z_size, kernel_size=1, stride=1),
            nn.Tanh(),
            nn.BatchNorm1d(z_size),
            
            nn.Conv1d(z_size, z_size*2, kernel_size=1, stride=1),
            custom_nn.Transpose((1,2)),
        )

    def forward(self, input, z):
        
        input_with_z = z
        
        recon_out = self.z_recon_fc(input_with_z)
        
        recon_dist = output_to_dist(recon_out)
        return recon_dist

