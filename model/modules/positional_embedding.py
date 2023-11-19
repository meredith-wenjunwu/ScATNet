import torch
from torch import nn, Tensor
import math
from typing import Optional, Any


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x is [Batch x Crops x Feature_dim]

        n_crops = x.size(1)
        x = x + self.pe[:, :n_crops]
        return x

#
class ScalePosEncoding(nn.Module):
    def __init__(self, num_scales, d_model):
        super(ScalePosEncoding, self).__init__()

        scale_embedding = torch.ones(num_scales, d_model)
        for j in range(num_scales):
            scale_embedding[j, :] = j
        scale_embedding = scale_embedding.unsqueeze(0)
        self.register_buffer('scale_embedding', scale_embedding)

    def forward(self, x):
        bsz = x.shape[0]
        x = x + self.scale_embedding.repeat(bsz, 1, 1)

        return x

