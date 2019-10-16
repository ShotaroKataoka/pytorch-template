import os

import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter

class TensorboardSummary(object):
    """
    Setting for TensorboardX.
    You don't have to touch this code.
    
    You can see logs when you run
    'tensorboard --logdir="<tensorboard directory>"'
    in command line.
    """
    def __init__(self, directory):
        self.directory = directory

    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer
