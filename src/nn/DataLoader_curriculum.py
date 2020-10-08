from __future__ import print_function, division
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
from tqdm import trange
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.nn.DataLoader import DataLoader_cipher_binary
import numpy as np

class DataLoader_curriculum(Dataset):
    """"""

    def __init__(self, X, Y, device, old_net, catgeorie, args, train = True):
        """
        """
        self.args = args
        self.categorie_1 = []
        self.categorie_2 = []
        self.categorie_3 = []
        self.old_net = old_net
        self.X, self.Y = X, Y
        self.data_val_c = DataLoader_cipher_binary(self.X, self.Y, device)
        self.dataloader_val_c = DataLoader(self.data_val_c, batch_size=self.args.batch_size,
                                      shuffle=False, num_workers=self.args.num_workers)
        self.device = device
        self.train = train
        self.catgeorie = catgeorie
        self.t = Variable(torch.Tensor([0.5]))
        if self.train:
            print("START PREPROCESSING")
            self.oder_input()
        self.categorie_22 = self.categorie_1 + self.categorie_2


    def __len__(self):
        if self.train:
            if self.catgeorie == 1:
                return len(self.categorie_1)
            if self.catgeorie == 2:
                return len(self.categorie_22)
            if self.catgeorie == 3:
                return len(self.Y)
        else:
            return len(self.Y)


    def __getitem__(self, idx):
        if self.train:
            if self.catgeorie == 1:
                input_x, input_y = self.X[self.categorie_1[idx]], self.Y[self.categorie_1[idx]]
                return torch.tensor(input_x).float(), torch.tensor(input_y).float()
            if self.catgeorie == 2:
                input_x, input_y = self.X[self.categorie_22[idx]], self.Y[self.categorie_22[idx]]
                return torch.tensor(input_x).float(), torch.tensor(input_y).float()
            if self.catgeorie == 3:
                input_x, input_y = self.X[idx], self.Y[idx]
                return torch.tensor(input_x).float(), torch.tensor(input_y).float()
        else:
            input_x, input_y = self.X[idx], self.Y[idx]
            return torch.tensor(input_x).float(), torch.tensor(input_y).float()



    def oder_input(self):
        self.old_net.eval()
        with torch.no_grad():
            for i, data in tqdm(enumerate(self.dataloader_val_c, 0)):
            #for index in trange(len(self.X)):
                inputs, labels = data
                offset = i*self.args.batch_size
                indexes = np.array([offset + nbre for nbre in range(self.args.batch_size)])
                # inputs, labels = torch.tensor(self.X[index]).float().to(self.device), torch.tensor(self.Y[index]).float().to(self.device)
                outputs = self.old_net(inputs.to(self.device))
                preds = (outputs.squeeze(1) > self.t.to(self.device)).float().cpu() * 1
                TP_index = (preds.eq(1) & labels.eq(1)).cpu()

                self.t2 = Variable(torch.Tensor([0.8]))
                preds2 = (outputs.squeeze(1) > self.t2.to(self.device)).float().cpu() * 1
                TP_index2 = (preds2.eq(1) & labels.eq(1)).cpu()
                A = indexes[TP_index]
                B = indexes[TP_index2]
                self.categorie_1 += np.intersect1d(A, B).tolist()
                self.t3 = Variable(torch.Tensor([0.6]))
                preds3a = (outputs.squeeze(1) > self.t3.to(self.device)).float().cpu() * 1
                preds3b = (outputs.squeeze(1) < self.t2.to(self.device)).float().cpu() * 1

                TP_index3 = (preds3a.eq(1) & preds3b.eq(1)).cpu()
                A = indexes[TP_index]
                B = indexes[TP_index3]
                self.categorie_2 += np.intersect1d(A, B).tolist()

                TN_index = (preds.eq(0) & labels.eq(0)).cpu()
                self.t2 = Variable(torch.Tensor([0.2]))
                preds2 = (outputs.squeeze(1) < self.t2.to(self.device)).float().cpu() * 1
                TP_index2 = (preds2.eq(1) & labels.eq(0)).cpu()
                A = indexes[TN_index]
                B = indexes[TP_index2]
                self.categorie_1 += np.intersect1d(A, B).tolist()
                self.t3 = Variable(torch.Tensor([0.4]))
                preds3a = (outputs.squeeze(1) < self.t3.to(self.device)).float().cpu() * 1
                preds3b = (outputs.squeeze(1) > self.t2.to(self.device)).float().cpu() * 1

                TP_index3 = (preds3a.eq(1) & preds3b.eq(1)).cpu()
                A = indexes[TP_index]
                B = indexes[TP_index3]
                self.categorie_2 += np.intersect1d(A, B).tolist()

        print()
        print("Nbre input categorie_1:", len(self.categorie_1))
        print("Nbre input categorie_2:", len(self.categorie_2))
        print("Nbre input categorie_3:", len(self.categorie_3))
        print("Nbre input all:", len(self.categorie_1)+len(self.categorie_2)+len(self.categorie_3))
        print()




#data_train = DataLoader_speck()
#print(data_train.X.shape, data_train.Y.shape)
