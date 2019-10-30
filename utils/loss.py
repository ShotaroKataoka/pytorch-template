import torch.nn as nn

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(weight=None, 
                                             size_average=None, 
                                             ignore_index=-100, 
                                             reduce=None, 
                                             reduction='mean')
    
    def forward(self, output, target):
        return self.criterion(output, target)