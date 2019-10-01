import torch
import torch.nn as nn
import torch.nn.functional as F
from sub_module import Blocks

import sys
sys.path.append('..')
from config import Config

conf = Config()

# Same Padding
def fixed_padding(inputs, kernel_size):
    kernel_size_effective = kernel_size + (kernel_size - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs

class Modeling(nn.Module):
    # 重みの定義などを行う。
    def __init__(self, c_in=conf.input_channel, c_out=conf.output_channel, c_hidden=conf.hidden_channel, hidden_layer=conf.hidden_layer, kernel_size=3):
        super(Modeling, self).__init__()
        self.kernel_size = kernel_size
        
        # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.fc_in = nn.Conv2d(c_in, c_hidden, kernel_size, stride=2)
        
        # sub_module.Blocks()
        self.blocks = Blocks(c_hidden, kernel_size, hidden_layer)
        
        # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv_out = nn.Conv2d(c_hidden, c_hidden*2, kernel_size, stride=2)
        
        # Predict weight
        self.fc = nn.Linear(c_hidden*2, c_out)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
        # 重み初期化
        self._init_weight()
    
    # モデルに入力xを与えたときに自動的に呼ばれる。出力を返す。
    def forward(self, x):
        # entry block
        x = fixed_padding(x, self.kernel_size)
        x = self.fc_in(x)
        x = self.relu(x)
        
        # sub_module.Blocks()
        x = self.blocks(x)
        
        # end block
        x = fixed_padding(x, self.kernel_size)
        x = self.conv_out(x)
        x = self.relu(x)
        
        # Global Average Pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        
        # predict
        x = self.fc(x.reshape((x.shape[0], x.shape[1])))
        return self.softmax(x)
    
    # 重みの初期化を行う。
    def _init_weight(self):
        for m in self.modules():
            # 全結合層なら
            if isinstance(m, nn.Linear):
                # Heの初期値で初期化
                torch.nn.init.kaiming_normal_(m.weight)
            # 2D畳み込み層なら
            elif isinstance(m, nn.Conv2d):
                # Heの初期値で初期化
                torch.nn.init.kaiming_normal_(m.weight)
    