import torch
from torch import nn
import numpy as np
from copy import deepcopy

class LSPro(nn.Module):       
    def __init__(self, feat_dim,hidden_size=2, weights=None):
        super(LSPro, self).__init__()
        self.reduced = nn.Linear(feat_dim,hidden_size,bias=False)
        self.reconstructed = nn.Linear(hidden_size,feat_dim,bias=False)
        nn.init.xavier_uniform_(self.reduced.weight)
        nn.init.xavier_uniform_(self.reconstructed.weight)

        if weights is not None:
            self.load_state_dict(deepcopy(weights))

    def forward(self, X,Y, train=False):
        reduced1 = (self.reduced(X))
        reduced2 = (self.reduced(Y))
        reconstructed1 = torch.sigmoid(self.reconstructed(reduced1))
        reconstructed2 = torch.sigmoid(self.reconstructed(reduced2))
    
        return [X,Y,reconstructed1,reconstructed2,reduced1,reduced2]