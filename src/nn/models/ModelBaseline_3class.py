import torch
import torch.nn as nn
from torch.nn import functional as F




class ModelPaperBaseline_3class(nn.Module):

    def __init__(self, args):
        super(ModelPaperBaseline_3class, self).__init__()
        self.args = args
        self.word_size = args.word_size
        self.conv0 = nn.Conv1d(in_channels=len(self.args.inputs_type), out_channels=16, kernel_size=1)
        self.BN0 = nn.BatchNorm1d(16, eps=0.01, momentum=0.99)
        self.layers_conv = nn.ModuleList()
        self.layers_batch = nn.ModuleList()
        self.numLayers = 10
        for i in range(10 - 1):
            self.layers_conv.append(nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, padding=1))
            self.layers_batch.append(nn.BatchNorm1d(16, eps=0.01, momentum=0.99))
        self.fc1 = nn.Linear(16 * args.word_size, 64)  # 6*6 from image dimension
        self.BN5 = nn.BatchNorm1d(64, eps=0.01, momentum=0.99)
        self.fc2 = nn.Linear(64, 64)
        self.BN6 = nn.BatchNorm1d(64, eps=0.01, momentum=0.99)
        self.fc3 = nn.Linear(64, 3)


    def forward(self, x):
        x = x.view(-1, len(self.args.inputs_type), self.word_size)
        self.x_input = x
        x = F.relu(self.BN0(self.conv0(x)))
        shortcut = x.clone()
        self.shorcut = shortcut[0]
        self.x_dico = {}
        for i in range(len(self.layers_conv)):
            x = self.layers_conv[i](x)
            x = self.layers_batch[i](x)
            x = F.relu(x)
            x = x + shortcut
            self.x_dico[i] = x
        x = x.view(x.size(0), -1)
        x = F.relu(self.BN5(self.fc1(x)))
        self.intermediare = x.clone()
        x = F.relu(self.BN6(self.fc2(x)))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        #x = torch.sigmoid(x)
        return x

    def freeze(self):
        self.conv0.weight.requires_grad = False
        self.BN0.bias.requires_grad = False
        for i in range(self.numLayers - 1):
            self.layers_conv[i].weight.requires_grad = False
            self.layers_batch[i].weight.requires_grad = False



