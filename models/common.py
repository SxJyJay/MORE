import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class FF(nn.Module):
    def __init__(self, num_units=[2048, 512]):
        super(FF, self).__init__()
        self.ff1 = nn.Conv1d(num_units[1], num_units[0], kernel_size=1)
        self.ff2 = nn.Conv1d(num_units[0], num_units[1], kernel_size=1)
        self.ln = nn.LayerNorm(num_units[1], eps=1e-8)

    def init_weights(self):
        pass

    def forward(self, inputs):
        outputs = F.relu(self.ff1(inputs.permute(0,2,1)))
        outputs = self.ff2(outputs).permute(0,2,1)
        outputs += inputs
        outputs = self.ln(outputs)
        return outputs



class MultiHead(nn.Module):
    def __init__(self, q_dim, k_dim,
                num_units=None,
                num_heads=8,
                dropout=0.1,
                post='111'):
        super(MultiHead, self).__init__()

        self.nh = num_heads
        if num_units is None:
            num_units = q_dim

        self.Q_p = nn.Linear(q_dim, num_units, bias=True)
        self.K_p = nn.Linear(k_dim, num_units, bias=True)
        self.V_p = nn.Linear(k_dim, num_units, bias=True)
        self.drop = nn.Dropout(dropout)

        self.post = post
        if self.post[0] == '1':
            self.ln = nn.LayerNorm(q_dim, eps=1e-8)

        if self.post[1] == '1':
            self.ff = FF([num_units*4, num_units])

        if self.post[2] == '1':
            self.proj = nn.Linear(num_units, num_units)

    def forward(self, queries):
        keys = queries
        num_heads = self.nh
        Q = F.relu(self.Q_p(queries))
        K = F.relu(self.K_p(keys))
        V = F.relu(self.V_p(keys))

        Q_ = torch.cat(torch.chunk(Q, num_heads, dim=2), dim=0)
        K_ = torch.cat(torch.chunk(K, num_heads, dim=2), dim=0)
        V_ = torch.cat(torch.chunk(V, num_heads, dim=2), dim=0)

        outputs = torch.bmm(Q_, K_.permute(0,2,1))
        outputs = outputs / (K_.size(2) ** 0.5)

        outputs = F.softmax(outputs, dim=2)
        outputs = self.drop(outputs)

        outputs = torch.bmm(outputs, V_)
        outputs = torch.cat(torch.chunk(outputs, num_heads, dim=0), dim=2)

        outputs += queries
        if self.post[0] == '1':
            outputs = self.ln(outputs)
        if self.post[1] == '1':
            outputs = self.ff(outputs)
        if self.post[2] == '1':
            outputs = self.proj(outputs)
        return outputs


