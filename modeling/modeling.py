import torch
import torch.nn as nn
import torch.nn.functional as F
from sub_module import HiddenLayer

import sys
sys.path.append('..')
from config import Config

conf = Config()

class Modeling(nn.Module):
    # 重みの定義などを行う。
    def __init__(self, c_in=conf.input_channel, c_out=conf.output_channel, c_hidden=conf.hidden_channel, hidden_layer=conf.hidden_layer):
        super(Modeling, self).__init__()
        
        self.fc_in = nn.Linear(c_in, c_hidden)
        self.hidden_module = HiddenLayer(c_hidden, hidden_layer)
        self.fc = nn.Linear(c_hidden, c_out)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)
        self._init_weight()
    
    # モデルに入力xを与えたときに自動的に呼ばれる。出力を返す。
    def forward(self, x):
        x = self.fc_in(x)
        x = self.relu(x)
        x = self.hidden_module(x)
        x = self.fc(x)
        return self.softmax(x)
    
    # 重みの初期化を行う。
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
    