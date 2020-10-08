from torch.nn import functional as F
import torch
import torch.nn as nn



class PermEqui1_mean(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(PermEqui1_mean, self).__init__()
    #self.Gamma = nn.Linear(in_dim, out_dim)
    self.conv0 = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=1)
    self.BN0 = nn.BatchNorm1d(out_dim, eps=0.01, momentum=0.99)



  def forward(self, x):
    xm = x.mean(1, keepdim=True)
    x = F.relu(self.BN0(self.conv0(x -xm)))

    return x




class DTanh(nn.Module):

  def __init__(self, args):
    super(DTanh, self).__init__()
    self.args = args
    self.d_dim = self.args.out_channel0
    self.x_dim = len(self.args.inputs_type)

    self.PermEqui1_mean1 = PermEqui1_mean(self.x_dim, self.d_dim)
    self.PermEqui1_mean2 = PermEqui1_mean(self.d_dim, self.d_dim)
    self.PermEqui1_mean3 = PermEqui1_mean(self.d_dim, self.d_dim)


    self.ro = nn.Sequential(
       #nn.Dropout(p=0.6),
       nn.Linear(16, 64),
       nn.Tanh(),
       #nn.Dropout(p=0.6),
       nn.Linear(64, 64),
        nn.Tanh(),
        # nn.Dropout(p=0.6),
        nn.Linear(64, 1),
    )



  def forward(self, x, test = "test.txt"):
    x = x.view(-1, len(self.args.inputs_type), self.args.word_size)
    x = torch.tanh(self.PermEqui1_mean1(x))
    shortcut = x.clone()
    x = torch.tanh(self.PermEqui1_mean2(x))
    x = x + shortcut
    x = torch.tanh(self.PermEqui1_mean3(x))
    x = x + shortcut
    self.intermediare = x.clone()
    sum_output, _ = x.max(1)
    ro_output = self.ro(sum_output)

    return torch.sigmoid(ro_output)

  def freeze(self):
      self.PermEqui1_mean1.conv0.weight.requires_grad = False
      self.PermEqui1_mean1.BN0.bias.requires_grad = False
      self.PermEqui1_mean2.conv0.weight.requires_grad = False
      self.PermEqui1_mean2.BN0.bias.requires_grad = False
      self.PermEqui1_mean3.conv0.weight.requires_grad = False
      self.PermEqui1_mean3.BN0.bias.requires_grad = False



def clip_grad(model, max_norm):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm ** 2
    total_norm = total_norm ** (0.5)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in model.parameters():
            p.grad.data.mul_(clip_coef)
    return total_norm
