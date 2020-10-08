import copy
import torch
from torch.utils.data import DataLoader
from src.nn.DataLoader import DataLoader_cipher_binary
from src.nn.DataLoader_curriculum import DataLoader_curriculum
from src.nn.models.ModelBaseline import ModelPaperBaseline
import time
from tqdm import tqdm
import os
import torch.nn as nn
from torch.autograd import Variable

from src.nn.models.Modelbaseline_CNN_ATTENTION import Modelbaseline_CNN_ATTENTION
from src.nn.models.Multi_Headed import Multihead
from src.nn.models.deepset import DTanh
from src.utils.utils import F1_Loss
from scipy.stats import sem
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Activation
import numpy as np
from pickle import dump

#NN
from tqdm import tqdm

def cyclic_lr(num_epochs, high_lr, low_lr):
    res = lambda i: low_lr + ((num_epochs - 1) - i % num_epochs) / (num_epochs - 1) * (high_lr - low_lr);
    return (res);


def make_checkpoint(datei):
    res = ModelCheckpoint(datei, monitor='val_loss', save_best_only=True);
    return (res);

def make_classifier(input_size=84, d1=128, d2=64, final_activation='sigmoid'):
    # Input and preprocessing layers
    inp = Input(shape=(input_size,));
    dense1 = Dense(d1)(inp);
    dense1 = BatchNormalization()(dense1);
    dense1 = Activation('relu')(dense1);
    out = Dense(1, activation=final_activation)(dense1);
    model = Model(inputs=inp, outputs=out);
    return (model);

def make_classifier2(input_size=84, d1=512, d2=256, final_activation='sigmoid'):
    # Input and preprocessing layers
    inp = Input(shape=(input_size,));
    dense1 = Dense(d1)(inp);
    dense1 = BatchNormalization()(dense1);
    dense1 = Activation('relu')(dense1);

    dense1bis = Dense(d1)(dense1);
    dense1bis = BatchNormalization()(dense1bis);
    dense1bis = Activation('relu')(dense1bis);

    dense1ter = Dense(d1)(dense1bis);
    dense1ter = BatchNormalization()(dense1ter);
    dense1ter = Activation('relu')(dense1ter);

    dense2bis = Dense(d2)(dense1ter);
    dense2bis = BatchNormalization()(dense2bis);
    dense2bis = Activation('relu')(dense2bis);

    dense2 = Dense(d2)(dense2bis);
    dense2 = BatchNormalization()(dense2);
    dense2 = Activation('relu')(dense2);
    out = Dense(1, activation=final_activation)(dense2);
    model = Model(inputs=inp, outputs=out);
    return (model);

def train_speck_distinguisher(args, n_feat, X, Y, X_eval, Y_eval, epoch, bs, name_ici="", wdir= "./", flag_3layes=False):
    # create the network
    if flag_3layes:
        net = make_classifier(input_size=n_feat);
    else:
        net = make_classifier2(input_size=n_feat);
    net.compile(optimizer='adam', loss='mse', metrics=['acc']);
    # generate training and validation data
    # set up model checkpoint
    check = make_checkpoint(wdir + 'NN_classifier' + str(args.nombre_round_eval) + "_"+ name_ici + '.h5');
    # create learnrate schedule
    lr = LearningRateScheduler(cyclic_lr(10, 0.0002, 0.0001));
    # train and evaluate
    h = net.fit(X, Y, epochs=epoch, batch_size=bs, validation_data=(X_eval, Y_eval), callbacks=[lr, check]);
    np.save(wdir + 'h_acc_' + str(np.max(h.history['val_acc'])) + "_"+ name_ici +  '.npy', h.history['val_acc']);
    np.save(wdir + 'h_loss' + str(args.nombre_round_eval) + "_"+ name_ici + '.npy', h.history['val_loss']);
    dump(h.history, open(wdir + 'hist' + str(args.nombre_round_eval) + "_"+ name_ici +  '.p', 'wb'));
    print("Best validation accuracy: ", np.max(h.history['val_acc']));
    if flag_3layes:
        net3 = make_classifier(input_size=n_feat);
    else:
        net3 = make_classifier2(input_size=n_feat);
    net3.load_weights(wdir + 'NN_classifier' + str(args.nombre_round_eval) + "_"+ name_ici +  '.h5')
    return (net3, h);


def train_speck_distinguisher2(args, n_feat, X, Y, X_eval, Y_eval, epoch, bs, name_ici="", wdir= "./", flag_3layes=False):
    # create the network
    net = make_classifier3(input_size=n_feat);

    net.compile(optimizer='adam', loss='mse', metrics=['acc']);
    # generate training and validation data
    # set up model checkpoint
    check = make_checkpoint(wdir + 'NN_classifier' + str(args.nombre_round_eval) + "_"+ name_ici + '.h5');
    # create learnrate schedule
    lr = LearningRateScheduler(cyclic_lr(10, 0.002, 0.001));
    # train and evaluate
    h = net.fit(X, Y, epochs=epoch, batch_size=bs, validation_data=(X_eval, Y_eval), callbacks=[lr, check]);
    np.save(wdir + 'h_acc_' + str(np.max(h.history['val_acc'])) + "_"+ name_ici +  '.npy', h.history['val_acc']);
    np.save(wdir + 'h_loss' + str(args.nombre_round_eval) + "_"+ name_ici + '.npy', h.history['val_loss']);
    dump(h.history, open(wdir + 'hist' + str(args.nombre_round_eval) + "_"+ name_ici +  '.p', 'wb'));
    print("Best validation accuracy: ", np.max(h.history['val_acc']));
    net3 = make_classifier3(input_size=n_feat);
    net3.load_weights(wdir + 'NN_classifier' + str(args.nombre_round_eval) + "_"+ name_ici +  '.h5')
    return (net3, h);

def make_classifier3(input_size=84, d1=64, d2=64, final_activation='sigmoid'):
    # Input and preprocessing layers
    inp = Input(shape=(input_size,));
    dense1 = Dense(d1)(inp);
    dense1 = BatchNormalization()(dense1);
    dense1 = Activation('relu')(dense1);
    dense2 = Dense(d2)(dense1);
    dense2 = BatchNormalization()(dense2);
    dense2 = Activation('relu')(dense2);
    out = Dense(1, activation=final_activation)(dense2);
    model = Model(inputs=inp, outputs=out);
    return (model);