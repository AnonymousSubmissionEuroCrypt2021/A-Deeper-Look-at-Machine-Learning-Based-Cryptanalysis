import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

import math

#DoReFaNet

class AE_binarize(nn.Module):

    # 99.5 %   ->64 float
    # 95 %   ->32 float
    # 88.5%     ->64 int
    # 87%   ->64 ->64 int
    # 96.6%  ->64 ->32 float
    # 93 %   ->64 ->18 float
    # 96.0%  ->64 ->32 ->32 float

    def __init__(self, args, input_sizze, h1 = 32, h2 = 64, h3 = 32, h4 = 128, h5 = 64, h6 = 32, h7 = 16):
        super(AE_binarize, self).__init__()
        self.args = args
        self.act_q = activation_quantize_fn(a_bit=1)
        self.fc1 = nn.Linear(input_sizze, h1)  # 6*6 from image dimension
        self.BN5 = nn.BatchNorm1d(h1, eps=0.01, momentum=0.99)
        #self.fc2 = nn.Linear(h1, h2)
        #self.BN6 = nn.BatchNorm1d(h2, eps=0.01, momentum=0.99)
        #self.fc3 = nn.Linear(h2, h3)
        #self.BN7 = nn.BatchNorm1d(h3, eps=0.01, momentum=0.99)
        #self.fc3b = nn.Linear(h3, h4)
        #self.BN7b = nn.BatchNorm1d(h4, eps=0.01, momentum=0.99)
        #self.fc3c = nn.Linear(h4, h5)
        #self.BN7c = nn.BatchNorm1d(h5, eps=0.01, momentum=0.99)
        #self.fc3d = nn.Linear(h5, h6)
        #self.BN7d = nn.BatchNorm1d(h6, eps=0.01, momentum=0.99)
        #self.fc3e = nn.Linear(h6, h7)
        #self.BN7e = nn.BatchNorm1d(h7, eps=0.01, momentum=0.99)

        #self.fc4a000 = nn.Linear(h7, h6)  # 6*6 from image dimension
        #self.BN8a000 = nn.BatchNorm1d(h6, eps=0.01, momentum=0.99)
        #self.fc4a00 = nn.Linear(h6, h5)  # 6*6 from image dimension
        #self.BN8a00 = nn.BatchNorm1d(h5, eps=0.01, momentum=0.99)
        #self.fc4a0 = nn.Linear(h5, h4)  # 6*6 from image dimension
        #self.BN8a0 = nn.BatchNorm1d(h4, eps=0.01, momentum=0.99)
        #self.fc4a = nn.Linear(h4, h3)  # 6*6 from image dimension
        #self.BN8a = nn.BatchNorm1d(h3, eps=0.01, momentum=0.99)
        #self.fc4 = nn.Linear(h3, h2)  # 6*6 from image dimension
        #self.BN8 = nn.BatchNorm1d(h2, eps=0.01, momentum=0.99)
        #self.fc5 = nn.Linear(h2, h1)
        #self.BN9 = nn.BatchNorm1d(h1, eps=0.01, momentum=0.99)
        self.fc6 = nn.Linear(h1, input_sizze)
        self.BN10 = nn.BatchNorm1d(input_sizze, eps=0.01, momentum=0.99)

        self.fc_classifiy = nn.Linear(input_sizze, 1024)
        self.BN_classifiy = nn.BatchNorm1d(1024, eps=0.01, momentum=0.99)
        self.fc_classifiy1 = nn.Linear(1024, 512)
        self.BN_classifiy1 = nn.BatchNorm1d(512, eps=0.01, momentum=0.99)
        #self.fc_classifiy2 = nn.Linear(h2, h3)
        #self.BN_classifiy2 = nn.BatchNorm1d(h3, eps=0.01, momentum=0.99)
        self.fc_classifiy3 = nn.Linear(512, 1)

        #self.fc_classifiy2 = nn.Linear(h1, 1)

        #self.fc_classifiy3 = nn.Linear(h1, 1)

    def classify(self, x):
        #x = self.embedding
        x = F.relu(self.BN_classifiy(self.fc_classifiy(x)))
        x = self.act_q(x)
        x = F.relu(self.BN_classifiy1(self.fc_classifiy1(x)))
        x = self.act_q(x)
        #x = F.relu(self.BN_classifiy2(self.fc_classifiy2(x)))
        x = self.fc_classifiy3(x)
        x = torch.sigmoid(x)
        return x

    def encoder(self, x):
        x =  self.BN5(self.fc1(x))
        #x = self.BN6(self.fc2(x))
        #x = self.BN7(self.fc3(x))
        #x = F.relu(self.BN7b(self.fc3b(x)))
        #x = F.relu(self.BN7c(self.fc3c(x)))
        #x = F.relu(self.BN7d(self.fc3d(x)))
        #x = F.relu(self.BN7e(self.fc3e(x)))

        return x

    def decoder(self, x):
        #x = F.relu(self.BN8a000(self.fc4a000(x)))
        #x = F.relu(self.BN8a00(self.fc4a00(x)))
        #x = F.relu(self.BN8a0(self.fc4a0(x)))
        #x = F.relu(self.BN8a(self.fc4a(x)))
        #x = F.relu(self.BN8(self.fc4(x)))
        #x = F.relu(self.BN9(self.fc5(x)))
        x = self.BN10(self.fc6(x))
        return x

    def forward(self, x):
        x = self.encoder(x)
        #x = self.act_q(x)
        self.embedding = x
        x = self.decoder(x)
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


