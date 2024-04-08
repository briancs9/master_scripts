import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, reduce, repeat
from torch.utils.data import Dataset, DataLoader
from torch.nn import Module, ModuleList, Linear, Dropout, LayerNorm, Identity, Parameter, init
from torchvision.transforms import v2



class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, padding: int, kernel_size=(7,1)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              stride=1,
                              kernel_size=kernel_size,
                              padding='same',
                              bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=(2,1),
                               padding=padding,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.skip = nn.Conv2d(in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=1,
                             stride=1,
                             padding='same',
                             bias=False)
        self.mp = nn.MaxPool2d(kernel_size=(2,1), padding=0)
        
    def forward(self, x):
        identity = x
        identity = self.skip(identity)
        identity = self.mp(identity)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x + identity)
        
        return x



class TempAxis(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(Block(1, 16, padding=(3,0)), 
                                   Block(16,16, padding=(3,0)), 
                                   Block(16,16, padding=(3,0)),
                                   nn.Dropout(0.2),
                                   Block(16,32, padding=(2,0), kernel_size=(5,1)),
                                   Block(32,32, padding=(2,0), kernel_size=(5,1)),
                                   Block(32,32, padding=(2,0), kernel_size=(5,1)),
                                   nn.Dropout(0.2),
                                   Block(32,64, padding=(1,0), kernel_size=(3,1)),
                                   Block(64,64, padding=(1,0), kernel_size=(3,1)),
                                   Block(64,64, padding=(1,0), kernel_size=(3,1)),
                                   nn.Dropout(0.2))
        self.pad = nn.ZeroPad2d((0,0,60,60))

    def sequence_length(self, n_channels=1, height=5000, width=8):
        return self.forward(torch.zeros((1, n_channels, height, width))).shape[1]
    
    def forward(self, x):
        x = self.pad(x)
        x = self.layer(x)
        
        return x
    


class LeadAxis(nn.Module):
    def __init__(self):
        super().__init__()
        self.lead_axis = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1,8)),
                                       nn.Dropout(0.2),
                                       nn.Flatten(),
                                       nn.Linear(1280,320),
                                       nn.Linear(320, 1))
        
    def forward(self, x):
        x = self.lead_axis(x)
        
        return x
 
    
class ECG_Model(nn.Module):
    def __init__(self, transformer_head=False):
        super().__init__()
        
        self.tokenizer = TempAxis()
        self.lead_axis = LeadAxis()
        self.transformer_head = transformer_head
        self.te_head = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1,8)),
                                     nn.Dropout(0.2),
                                     nn.Flatten(2,3))
        self.te = nn.TransformerEncoderLayer(d_model=128, nhead=2, batch_first=True)
        self.flat = nn.Flatten()
        self.linear = nn.Sequential(nn.Linear(1280, 320),
                                    nn.Linear(320, 1))
    
    def forward(self, x):
        x = self.tokenizer(x)
        
        if self.transformer_head:
            x = self.te_head(x)
            x = x.view(-1, 10, 128)
            x = self.te(x)
            x = self.flat(x)
            x = self.linear(x)
        else:
            x = self.lead_axis(x)
        return x
    

