import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling.sub_module import Blocks, PoolBlock
sys.path.append('..')
from config import Config

# instance of config
conf = Config()

class Modeling(nn.Module):
    """
    This module is main definition of your model.
    
    __init__(): Define weights of model.
    forward(): Compute outputs from inputs.
    _init_weight(): Initialize weights value of model.
    
    Blocks, PoolBlock: Sub_module of this model.
                       You can change these in <modeling.sub_module>.
    """
    def __init__(self, c_in=conf.input_channel, c_out=conf.num_class, c_hidden=conf.hidden_channel, hidden_layer=conf.hidden_layer, kernel_size=3):
        super(Modeling, self).__init__()
        self.kernel_size = kernel_size
        
        # Define sub_module.PoolBlock()
        self.conv_pool1 = PoolBlock(c_in, kernel_size, is_first=True)
        self.conv_pool2 = PoolBlock(c_hidden, kernel_size)
        
        # Define sub_module.Blocks()
        self.blocks1 = Blocks(c_hidden, kernel_size, hidden_layer)
        self.blocks2 = Blocks(c_hidden*2, kernel_size, hidden_layer)
        
        # Define last layer.
        self.fc = nn.Linear(c_hidden*2, c_out)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
        # Initialize weights.
        self._init_weight()
    
    def forward(self, x):
        x = self.conv_pool1(x)
        x = self.blocks1(x)
        x = self.conv_pool2(x)
        x = self.blocks2(x)
        
        # Global Average Pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        
        # predict
        x = self.fc(x.reshape((x.shape[0], x.shape[1])))
        return self.softmax(x)
    
    def _init_weight(self):
        for m in self.modules():
            # if fully connected layer
            if isinstance(m, nn.Linear):
                # initialize value (He)
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
    