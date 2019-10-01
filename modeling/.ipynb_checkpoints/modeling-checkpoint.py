import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sub_module import Blocks, PoolBlock

from config import Config
conf = Config()

class Modeling(nn.Module):
    # 重みの定義などを行う。
    def __init__(self, c_in=conf.input_channel, c_out=conf.num_class, c_hidden=conf.hidden_channel, hidden_layer=conf.hidden_layer, kernel_size=3):
        super(Modeling, self).__init__()
        self.kernel_size = kernel_size
        
        # sub_module.PoolBlock()
        self.conv_pool1 = PoolBlock(c_in, kernel_size, is_first=True)
        self.conv_pool2 = PoolBlock(c_hidden, kernel_size)
        
        # sub_module.Blocks()
        self.blocks1 = Blocks(c_hidden, kernel_size, hidden_layer)
        self.blocks2 = Blocks(c_hidden*2, kernel_size, hidden_layer)
        
        # Predict weight
        self.fc = nn.Linear(c_hidden*2, c_out)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
        # 重み初期化
        self._init_weight()
    
    # モデルに入力xを与えたときに自動的に呼ばれる。出力を返す。
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
    