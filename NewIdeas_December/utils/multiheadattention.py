import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class MultiHeadAttentionWithoutQ(nn.Module):
    def __init__(self, d_k, d_model, n_heads, dropout):
        super().__init__()

        # assume d_v = d_k
        self.d_k = d_k
        self.n_heads = n_heads
        self.dropout = dropout

        self.key_proj = nn.Linear(in_features=d_model, out_features=1)
        self.value_proj = nn.Linear(in_features=d_model, out_features=d_k * n_heads)
        ###___self.query_proj = nn.Linear(d_model, d_k * n_heads)

        #final Linear Layer
        self.finalFC = nn.Linear(d_k * n_heads, d_model)

    def forward(self, k, v):
        ###___q = self.query_proj(q) # batch * Seq * (hd_k)
        k = F.dropout(self.key_proj(k), p=self.dropout) # batch * Seq * (hd_k)
        v = F.dropout(self.value_proj(v), p=self.dropout) # batch * Seq * (hd_k)

        N = k.shape[0] #batch size in batch_first mode
        T = k.shape[1] #Sequence length


        #change the shape to:
        # (N, T, h, d_k) --> (N, h, T, d_k) in order to matmul to work correctly with dims
        ###___q = q.view(N, T, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(N, T, 1, 1).transpose(1, 2)
        v = v.view(N, T, self.n_heads, self.d_k).transpose(1, 2)

        #Compute attention weights: (N,h,T,dk)@(N,h,dk,T) = (N,h,T,T)
        ###___att_scores = q @ k.transpose(-2, -1) / sqrt(self.d_k)
        att_scores = k.transpose(-2, -1) / sqrt(self.d_k)
        att_weights = F.softmax(att_scores, dim=-1)

        #(N,h,T,T) @ (N,h,T,dk) = (N,h,T,dk)
        A = att_weights @ v

        A = A.contiguous().view(N, self.d_k * self.n_heads) #(N, T, h*dk)

        A = self.finalFC(A) #(N * d_model)

        return A, att_weights