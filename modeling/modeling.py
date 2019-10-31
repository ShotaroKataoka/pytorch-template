import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling.xception import Xception


class Modeling(nn.Module):
    """
    This module is main definition of your model.
    
    __init__(): Define weights of model.
    forward(): Compute outputs from inputs.
    _init_weight(): Initialize weights value of model.
    
    Xception: example of modeling program.
    """
    def __init__(self, num_classes):
        super(Modeling, self).__init__()
        
        self.xception = Xception(num_classes=num_classes)
        self.softmax = nn.Softmax()
        
        self._init_weight()
    
    def forward(self, x):
        x = self.xception(x)
        x = self.softmax(x)
        return x
    
    def _init_weight(self):
        """
        You have to define this, if you need.
        """
        pass
    