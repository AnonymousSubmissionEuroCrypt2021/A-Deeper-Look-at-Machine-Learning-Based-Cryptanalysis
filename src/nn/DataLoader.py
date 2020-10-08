from __future__ import print_function, division
import torch
from torch.utils.data import Dataset
import numpy as np


class DataLoader_cipher_binary(Dataset):
    """"""

    def __init__(self, X, Y, device):
        self.X, self.Y = X, Y
        self.device = device

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        input_x, input_y = self.X[idx], self.Y[idx]
        return torch.tensor(input_x).float(), torch.tensor(input_y).float()


class DataLoader_cipher_binaryNbatch(Dataset):
    """"""

    def __init__(self, X, Y, args, device):
        nbits = len(args.inputs_type)* args.word_size
        for batch in range(args.Nbatch):
            if batch == 0:
                Xfinal = X[:,batch*nbits:(batch+1)*nbits]
                Xfinal2 = Xfinal.reshape(-1,len(args.inputs_type),args.word_size,1)
            else:
                Xfinal = X[:,batch * nbits:(batch + 1) * nbits]
                Xfinal = Xfinal.reshape(-1, len(args.inputs_type), args.word_size, 1)
                Xfinal2 = np.concatenate((Xfinal2, Xfinal), axis=3)
        self.X, self.Y = Xfinal2, Y
        print(self.X.shape)

        self.device = device

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        input_x, input_y = self.X[idx], self.Y[idx]
        return torch.tensor(input_x).float(), torch.tensor(input_y).float()


