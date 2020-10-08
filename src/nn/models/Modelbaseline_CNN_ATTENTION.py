import torch
import torch.nn as nn
from torch.nn import functional as F
from src.nn.models.attention_augmented_conv import AugmentedConv



class Modelbaseline_CNN_ATTENTION(nn.Module):

    def __init__(self, args):
        super(Modelbaseline_CNN_ATTENTION, self).__init__()
        self.word_size = args.word_size
        self.args = args
        self.conv0 = AugmentedConv(in_channels=len(self.args.inputs_type), out_channels=self.args.out_channel0, dk=40, dv=4, Nh=4, kernel_size=1)
        self.BN0 = nn.BatchNorm2d(self.args.out_channel0, eps=0.01, momentum=0.99)
        self.layers_conv = nn.ModuleList()
        self.layers_batch = nn.ModuleList()
        self.numLayers = self.args.numLayers
        for i in range(self.args.numLayers - 1):
            self.layers_conv.append(AugmentedConv(in_channels=self.args.out_channel1, out_channels=self.args.out_channel1, kernel_size=3, dk=40, dv=4, Nh=4))
            self.layers_batch.append(nn.BatchNorm2d(self.args.out_channel1, eps=0.01, momentum=0.99))
        self.fc1 = nn.Linear(self.args.out_channel1 * self.args.word_size, self.args.hidden1)  # 6*6 from image dimension
        self.BN5 = nn.BatchNorm1d(self.args.hidden1, eps=0.01, momentum=0.99)
        self.fc2 = nn.Linear(self.args.hidden1, self.args.hidden1)
        self.BN6 = nn.BatchNorm1d(self.args.hidden1, eps=0.01, momentum=0.99)
        self.fc3 = nn.Linear(self.args.hidden1, 1)

    def forward(self, x):
        x = x.view(-1, len(self.args.inputs_type), self.word_size, 1)
        #x = x.view(-1, 2 * self.num_blocks, self.word_size, 1)
        x = F.relu(self.BN0(self.conv0(x)))
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
        x = torch.sigmoid(x)
        return x

    def freeze(self):
        self.conv0.weight.requires_grad = False
        self.BN0.bias.requires_grad = False
        for i in range(self.numLayers - 1):
            self.layers_conv[i].weight.requires_grad = False
            self.layers_batch[i].weight.requires_grad = False
