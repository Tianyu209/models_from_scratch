import torch
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
    def __init__(self,d_k,dropout=0.1):
        super().__init__()
        self.d_k =d_k
        self.dropout = dropout
    def forward(self, q, k, v, mask=None):
        #QK^T/d_k, rescale the variation to 1
        attn =torch.matmul(q/self.d_k,k.transpose(2,3))
        #mask attn
        if mask is not None:
            attn = attn.masked_fill(mask==0,-1e9)
        attn = self.dropout(nn.functional.softmax())
        res =torch.matmul(attn,v)
        return res,attn
