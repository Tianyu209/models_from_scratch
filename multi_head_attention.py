###
import torch
from torch import nn
import torch.functional as F
import math
from time import time
###
"""class implement"""
class multi_head_attention(nn.Module):
    def __init__(self,d_model,n_head):
        super(multi_head_attention,self).__init__()
        self.n_head  = n_head
        self.d_model = d_model
        """QKV"""
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.combine = nn.Linear(d_model,d_model)
        self.softmax = nn.Softmax(dim = -2)
    def forward(self,q,k,v):
        """init para"""
        batch, time,dim = q.shape
        n_d = self.d_model // self.n_head
        q,k,v = self.w_q(q),self.w_k(k),self.w_v(v)
        q = q.view(batch, time,self.n_head,n_d).permute(0,2,1,3)
        k = k.view(batch, time,self.n_head,n_d).permute(0,2,1,3)
        v = v.view(batch, time,self.n_head,n_d).permute(0,2,1,3)
        """Z = softmax((QK.T)/sqrt(n_d))"""
        z = q@ k.transpose(2,3)/math.sqrt(n_d)
        """mask"""
        mask = torch.tril(torch.ones(time,time,dtype=bool))
        z = z.masked_fill(mask == 0,float("-inf"))
        """Z = softmax((QK.T)/sqrt(n_d))@V"""
        z = self.softmax(z)@v
        z =z.permute(0,2,1,3).contiguous().view(batch,time,dim)

        final = self.combine(z)
        return final
if __name__ == "__main__":
    """random input (batch, time, em_dim)"""
    X = torch.randn(128,64,512)
    """basic para"""
    d_model =512 # QKV
    n_head =8 #head
    """time & test"""
    from time import time
    t = time()
    attention = multi_head_attention(d_model,n_head)
    output = attention(X,X,X)
    tt = time()
    print(output.shape)
    print("The running time is "+ str(tt-t))