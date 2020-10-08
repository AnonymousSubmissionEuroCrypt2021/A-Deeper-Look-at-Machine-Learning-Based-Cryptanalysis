import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

import math

#DoReFaNet

class ModelPaperBaseline_bin_bagnet(nn.Module):

    def __init__(self, args):
        super(ModelPaperBaseline_bin_bagnet, self).__init__()
        self.args = args
        self.word_size = args.word_size
        self.act_q = activation_quantize_fn(a_bit=1)
        self.convm1 = nn.Conv1d(in_channels=len(self.args.inputs_type), out_channels=len(self.args.inputs_type), kernel_size=1)
        self.BNm1 = nn.BatchNorm1d(len(self.args.inputs_type), eps=0.01, momentum=0.99)
        self.conv0 = nn.Conv1d(in_channels=len(self.args.inputs_type), out_channels=args.out_channel0, kernel_size=1)
        self.BN0 = nn.BatchNorm1d(args.out_channel0, eps=0.01, momentum=0.99)
        self.layers_conv = nn.ModuleList()
        self.layers_batch = nn.ModuleList()
        self.numLayers = args.numLayers
        for i in range(args.numLayers - 1):
            if i ==0:
                self.layers_conv.append(
                    nn.Conv1d(in_channels=args.out_channel0, out_channels=args.out_channel1, kernel_size=3, padding =1))
                self.layers_batch.append(nn.BatchNorm1d(args.out_channel1, eps=0.01, momentum=0.99))
            else:
                self.layers_conv.append(
                nn.Conv1d(in_channels=args.out_channel1, out_channels=args.out_channel1, kernel_size=1))
                self.layers_batch.append(nn.BatchNorm1d(args.out_channel1, eps=0.01, momentum=0.99))
        self.fc1 = nn.Linear(args.out_channel1 * 5, args.hidden1)  # 6*6 from image dimension
        self.BN5 = nn.BatchNorm1d(args.hidden1, eps=0.01, momentum=0.99)
        self.fc2 = nn.Linear(args.hidden1, args.hidden1)
        self.BN6 = nn.BatchNorm1d(args.hidden1, eps=0.01, momentum=0.99)
        self.fc3 = nn.Linear(args.hidden1, 1)
        self.fc4 = nn.Linear(12, 1)

    def forward(self, x):
        x = x.view(-1, len(self.args.inputs_type), self.word_size)
        for i in range(12):
            x2 = x[:,:,i:i+5]
            res = self.encode_patch(x2)
            if i ==0:
                x1 = res
            else:
                x1 = torch.cat((x1, res), dim=1)
        x = self.fc4(x1)
        x = torch.sigmoid(x)
        return x

    def encode_patch(self, x):

        x = F.relu(self.BNm1(self.convm1(x)))
        x = F.relu(self.BN0(self.conv0(x)))
        x = self.act_q(x)
        shortcut = x.clone()
        self.shorcut = shortcut
        for i in range(len(self.layers_conv)):
            x = self.layers_conv[i](x)
            x = self.layers_batch[i](x)
            x = F.relu(x)
            x = x + shortcut
        x = self.act_q(x)
        self.classify = x
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



def uniform_quantize(k):
  class qfn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
      if k == 32:
        out = input
      elif k == 1:
        out = torch.sign(input)
      else:
        n = float(2 ** k - 1)
        out = torch.round(input * n) / n
      return out

    @staticmethod
    def backward(ctx, grad_output):
      grad_input = grad_output.clone()
      return grad_input

  return qfn().apply


class weight_quantize_fn(nn.Module):
  def __init__(self, w_bit):
    super(weight_quantize_fn, self).__init__()
    assert w_bit <= 8 or w_bit == 32
    self.w_bit = w_bit
    self.uniform_q = uniform_quantize(k=w_bit)

  def forward(self, x):
    if self.w_bit == 32:
      weight_q = x
    elif self.w_bit == 1:
      E = torch.mean(torch.abs(x)).detach()
      weight_q = self.uniform_q(x / E) * E
    else:
      weight = torch.tanh(x)
      max_w = torch.max(torch.abs(weight)).detach()
      weight = weight / 2 / max_w + 0.5
      weight_q = max_w * (2 * self.uniform_q(weight) - 1)
    return weight_q


class activation_quantize_fn(nn.Module):
  def __init__(self, a_bit):
    super(activation_quantize_fn, self).__init__()
    assert a_bit <= 8 or a_bit == 32
    self.a_bit = a_bit
    self.uniform_q = uniform_quantize(k=a_bit)

  def forward(self, x):
    if self.a_bit == 32:
      activation_q = x
    else:
      activation_q = self.uniform_q(torch.clamp(x, 0, 1))
      # print(np.unique(activation_q.detach().numpy()))
    return activation_q


def conv1d_Q_fn(w_bit):
  class Conv1d_Q(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
      super(Conv1d_Q, self).__init__(in_channels, out_channels, kernel_size, stride,
                                     padding, dilation, groups, bias)
      self.w_bit = w_bit
      self.quantize_fn = weight_quantize_fn(w_bit=w_bit)

    def forward(self, input, order=None):
      self.weight_q = self.quantize_fn(self.weight)
      # print(np.unique(weight_q.detach().numpy()))
      #print(self.weight_q)
      return F.conv1d(input, self.weight_q, self.bias, self.stride,
                      self.padding, self.dilation, self.groups)

  return Conv1d_Q


def linear_Q_fn(w_bit):
  class Linear_Q(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
      super(Linear_Q, self).__init__(in_features, out_features, bias)
      self.w_bit = w_bit
      self.quantize_fn = weight_quantize_fn(w_bit=w_bit)

    def forward(self, input):
      weight_q = self.quantize_fn(self.weight)
      # print(np.unique(weight_q.detach().numpy()))
      return F.linear(input, weight_q, self.bias)

  return Linear_Q


