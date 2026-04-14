import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        B = q.size(0)

        Q = self.q(q).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k(k).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v(v).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, V)

        out = out.transpose(1, 2).contiguous().view(B, -1, self.num_heads * self.d_k)
        return self.fc(out)