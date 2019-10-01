import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('..')
from config import Config

def fixed_padding(inputs, kernel_size):
    kernel_size_effective = kernel_size + (kernel_size - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs

class Blocks(nn.Module):
    def __init__(self, c_hidden, kernel_size, hidden_layer):
        super(Blocks, self).__init__()
        self.kernel_size = kernel_size
        
        self.fc_hidden = nn.ModuleList([nn.Conv2d(c_hidden, c_hidden, kernel_size) for i in range(hidden_layer)])
        self.relu = nn.ReLU()
        self._init_weight()
        
    def forward(self, x):
        for f in self.fc_hidden:
            x = fixed_padding(x, self.kernel_size)
            x = f(x)
            x = self.relu(x)
        return x
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
    