#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2022 Muqiao Yang <muqiaoy@andrew.cmu.edu>

import numpy as np
import torch
import torchaudio
import torch.nn as nn
from torchaudio.pipelines import Wav2Vec2Bundle
from .Demucs.utils import capture_init

# class Egemaps_estimator(nn.Module):
#     @capture_init
#     def __init__(self, smile_F, fs=16000):
#         super().__init__()
#         self.smile_F = smile_F
#         self.fs = fs
#         self.out_dim = len(self.smile_F.feature_names)
#         assert self.out_dim == 88, "Not implemented for other dimensions"

#         bundle = torchaudio.pipelines.HUBERT_BASE
#         self.feature_extractor = bundle.get_model()
#         for param in self.feature_extractor.parameters():
#             param.requires_grad = False
#         self.estimator = nn.Sequential(
#                 nn.Linear(bundle._params['encoder_embed_dim'], 256),
#                 nn.ReLU(),
#                 nn.Linear(256, self.out_dim)
#                 )

#     def forward(self, waveform):
#         # bs, dim_1, Nx = waveform.shape
#         # assert dim_1 == 1
#         waveform = waveform.squeeze(dim=1)

#         features, _ = self.feature_extractor(waveform)
#         pooled = features.mean(dim=1)
#         egemaps_features = self.estimator(pooled)
#         return egemaps_features


class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """

    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)


    def forward(self, batch_rep):
        """
        input:
            batch_rep : size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension
        attention_weight:
            att_w : size (N, T, 1)
        return:
            utter_rep: size (N, H)
        """
        softmax = nn.functional.softmax
        att_w = softmax(self.W(batch_rep).squeeze(-1)).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)

        return utter_rep


