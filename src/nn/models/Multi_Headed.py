import torch
import torch.nn as nn
from torch.nn import functional as F

import random
import numpy as np

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Multihead(nn.Module):

    def __init__(self, args):
        super(Multihead, self).__init__()
        self.word_size = args.word_size
        self.args = args
        self.conv00 = nn.Conv1d(in_channels=len(self.args.inputs_type), out_channels=32, kernel_size=1)
        self.BN00 = nn.BatchNorm1d(32,  eps=0.01, momentum=0.99)
        self.conv01 = nn.Conv1d(in_channels=len(self.args.inputs_type), out_channels=32, kernel_size=2)
        self.BN01 = nn.BatchNorm1d(32, eps=0.01, momentum=0.99)
        self.conv02 = nn.Conv1d(in_channels=len(self.args.inputs_type), out_channels=32, kernel_size=3)
        self.BN02 = nn.BatchNorm1d(32, eps=0.01, momentum=0.99)
        self.conv03 = nn.Conv1d(in_channels=len(self.args.inputs_type), out_channels=32, kernel_size=4)
        self.BN03 = nn.BatchNorm1d(32, eps=0.01, momentum=0.99)
        self.layers_conv = nn.ModuleList()
        self.layers_batch = nn.ModuleList()
        self.numLayers = args.numLayers
        for i in range(args.numLayers - 1):
            self.layers_conv.append(nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1))
            self.layers_batch.append(nn.BatchNorm1d(32, eps=0.01, momentum=0.99))
        self.fc1 = nn.Linear(1856, 64)  # 6*6 from image dimension
        self.BN5 = nn.BatchNorm1d(64, eps=0.01, momentum=0.99)
        self.fc2 = nn.Linear(64, 64)
        self.BN6 = nn.BatchNorm1d(64, eps=0.01, momentum=0.99)
        self.fc3 = nn.Linear(64, 1)



    def forward(self, x, type_training="classic"):
        x = x.view(-1, len(self.args.inputs_type), self.word_size)
        x1 = F.relu(self.BN00(self.conv00(x)))
        x2 = F.relu(self.BN01(self.conv01(x)))
        x3 = F.relu(self.BN02(self.conv02(x)))
        x4 = F.relu(self.BN03(self.conv03(x)))
        x = torch.cat((x1, x2 ,x3 ,x4), dim=2)
        shortcut = x.clone()
        for i in range(len(self.layers_conv)):
            x = self.layers_conv[i](x)
            x = self.layers_batch[i](x)
            x = F.relu(x)
            x = x + shortcut
        x = x.view(x.size(0), -1)
        self.intermediare = x.clone()
        x = F.relu(self.BN5(self.fc1(x)))
        x = F.relu(self.BN6(self.fc2(x)))
        x = self.fc3(x)
        if type_training == "classic":
            x = torch.sigmoid(x)
        return x

    def freeze(self):
        self.conv00.weight.requires_grad = False
        self.BN00.bias.requires_grad = False
        self.conv01.weight.requires_grad = False
        self.BN01.bias.requires_grad = False
        self.conv02.weight.requires_grad = False
        self.BN02.bias.requires_grad = False
        self.conv03.weight.requires_grad = False
        self.BN03.bias.requires_grad = False
        for i in range(self.numLayers - 1):
            self.layers_conv[i].weight.requires_grad = False
            self.layers_batch[i].weight.requires_grad = False



