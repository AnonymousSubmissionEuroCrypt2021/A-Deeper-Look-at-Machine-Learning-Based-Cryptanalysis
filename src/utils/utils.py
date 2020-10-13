import argparse, os
import torch
import torch.nn as nn
from torch.nn import functional as F

def dir_path(string):
    if os.path.isfile(string):
        return string
    else:
        raise NotADirectoryError(string)

def str2list(v):
    if isinstance(v, str):
        return [float(v)]
    else:
        return v
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2hexa(x):
    try:
        x2 = x.replace(")", "").replace("(", "").split(", ")
        return int(x2[0], 16), int(x2[1], 16)
    except:
        return x

def two_args_str_int(x):
    try:
        return int(x)
    except:
        return x

def two_args_str_float(x):
    try:
        return float(x)
    except:
        return x

def transform_input_type(x):
    if "[" in x:
        inputs_type1 = []
        inputs_type2 = x.replace("[", "").replace("]", "")
        inputs_type1 = inputs_type2.split(", ")
        x = inputs_type1
    return x

def transform_input_type2(x):
    if "[" in x:
        inputs_type2 = x.replace("[", "").replace("]", "")
        inputs_type1 = inputs_type2.split(", ")
        x = inputs_type1
        x2 = [int(y) for y in x]
    return x2


class F1_Loss(nn.Module):
    '''Calculate F1 score. Can work with gpu tensors

    The original implmentation is written by Michal Haltuf on Kaggle.

    Returns
    -------
    torch.Tensor
        `ndim` == 1. epsilon <= val <= 1

    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    - http://www.ryanzhang.info/python/writing-your-own-loss-function-module-for-pytorch/
    '''

    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true, ):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true.long(), 2).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
        return 1 - f1.mean()



class BCE_bit_Loss(nn.Module):


    def __init__(self, lambda1 =1 ,lambda2=1, lambda3=1, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

    def f1(self, y_pred, y_true):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true.long(), 2).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
        return 1 - f1.mean()

    def bit_loss(self, y_pred, y_true ):
        return ((1 - y_pred) ** y_true + y_pred ** (1 - y_true)).mean()


    def forward(self, y_pred, y_true ):
        return self.lambda1 * F.mse_loss(y_pred, y_true) + self.lambda2 *self.f1(y_pred.unsqueeze(1), y_true) + self.lambda3 *self.bit_loss(y_pred, y_true)





class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive

def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.uniform_(m.weight)
