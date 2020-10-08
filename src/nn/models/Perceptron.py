import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

import math

#DoReFaNet
class Perceptron(nn.Module):
    def __init__(self, args):
        super(Perceptron, self).__init__()

        self.fc1 = nn.Linear(64, 1024)  # 6*6 from image dimension
        self.BN5 = nn.BatchNorm1d(1024, eps=0.01, momentum=0.99)
        self.fc2 = nn.Linear(1024, 512)
        self.BN6 = nn.BatchNorm1d(512, eps=0.01, momentum=0.99)
        self.fc3 = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.BN5(self.fc1(x)))
        self.intermediare = x
        x = F.relu(self.BN6(self.fc2(x)))

        x = self.fc3(x)

        x = torch.sigmoid(x)
        return x






import torch.nn as nn

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

class BinarizeLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=Binarize(self.weight.org)
        out = nn.functional.linear(input, self.weight)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)
        return out


def Binarize(tensor,quant_mode='det'):
    if quant_mode=='det':
        return tensor.sign()
    else:
        return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0,1).round().mul_(2).add_(-1)
