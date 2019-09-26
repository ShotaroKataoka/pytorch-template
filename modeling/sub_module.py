import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('..')
from config import Config

class HiddenLayer(nn.Module):
    def __init__(self, c_hidden, hidden_layer):
        super(HiddenLayer, self).__init__()
        self.fc_hidden = nn.ModuleList([nn.Linear(c_hidden, c_hidden) for i in range(hidden_layer)])
        self.relu = nn.ReLU()
        self._init_weight()
        
    def forward(self, x):
        for f in self.fc_hidden:
            x = f(x)
            x = self.relu(x)
        return x
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
    