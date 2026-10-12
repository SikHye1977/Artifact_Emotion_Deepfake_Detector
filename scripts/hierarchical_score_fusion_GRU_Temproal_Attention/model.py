"""Embedding + native emotion probabilities -> GRU -> masked temporal attention."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .common import DIMS, CLASSES

class TokenBottleneck(nn.Module):
    def __init__(self,dim):
        super().__init__();self.linear=nn.Linear(dim,128);self.bn=nn.BatchNorm1d(128)
        self.activation=nn.ReLU();self.dropout=nn.Dropout(.2)
    def forward(self,x):
        x=self.linear(x)
        # A one-token training batch has no meaningful batch variance. Use running stats only in this case.
        if self.training and len(x)==1:
            x=nn.functional.batch_norm(x,self.bn.running_mean,self.bn.running_var,self.bn.weight,self.bn.bias,False,0.,self.bn.eps)
        else:x=self.bn(x)
        return self.dropout(self.activation(x))

class EmotionFlowHead(nn.Module):
    def __init__(self,modality):
        super().__init__();self.modality=modality
        hidden,layers=(64,1) if modality=='video' else (128,2)
        self.projection=TokenBottleneck(DIMS[modality])
        self.gru=nn.GRU(128+len(CLASSES[modality]),hidden,num_layers=layers,batch_first=True,
                        bidirectional=False,dropout=.2 if layers>1 else 0.)
        self.attention=nn.Linear(hidden,1)
        self.classifier=nn.Sequential(nn.Linear(hidden,64),nn.ReLU(),nn.Dropout(.2),nn.Linear(64,1))
    def forward(self,embedding,probabilities,mask):
        if embedding.ndim!=3 or mask.shape!=embedding.shape[:2] or mask.dtype!=torch.bool:
            raise ValueError('Expected [B,T,D] and boolean prefix mask')
        lengths=mask.sum(1)
        expected=torch.arange(mask.shape[1],device=mask.device)[None,:]<lengths[:,None]
        if not torch.equal(mask,expected) or not bool((lengths>0).all()):
            raise ValueError('Compact valid tokens in timestamp order; exclude all-missing samples')
        if probabilities.shape!=(*mask.shape,len(CLASSES[self.modality])):
            raise ValueError('Unexpected emotion probability dimension')
        e,p=embedding[mask],probabilities[mask]
        if not bool(torch.isfinite(e).all()) or not bool(torch.isfinite(p).all()):raise ValueError('Invalid valid token')
        if not bool(((p>=0)&(p<=1)).all()) or not torch.allclose(p.sum(-1),torch.ones_like(p[:,0]),atol=1e-5):
            raise ValueError('Invalid native emotion probabilities')
        # Padding never enters BatchNorm, GRU, attention or a time-axis average.
        projected=self.projection(e)
        joined=torch.cat([projected,p],dim=-1)
        x=joined.new_zeros((*mask.shape,joined.shape[-1]));x[mask]=joined
        packed=pack_padded_sequence(x,lengths.cpu(),batch_first=True,enforce_sorted=False)
        packed_h,_=self.gru(packed)
        h,_=pad_packed_sequence(packed_h,batch_first=True,total_length=mask.shape[1])
        alpha=torch.softmax(self.attention(h).squeeze(-1).masked_fill(~mask,-torch.inf),dim=1)
        context=(h*alpha.unsqueeze(-1)).sum(1)
        return self.classifier(context).squeeze(-1),alpha
