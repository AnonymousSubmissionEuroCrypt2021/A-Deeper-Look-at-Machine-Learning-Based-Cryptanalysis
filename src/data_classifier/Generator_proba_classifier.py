import numpy as np
from tqdm import tqdm
from scipy import sparse
#from sparse_vector import SparseVector
import time

class Genrator_data_prob_classifier:

    def __init__(self, args, net, path_file_models, rng, creator_data_binary, device, masks, nn_model_ref):

        self.args = args
        self.device= device
        self.net = net
        self.masks = masks
        self.path_file_models = path_file_models
        self.rng = rng
        self.creator_data_binary = creator_data_binary
        self.nn_model_ref = nn_model_ref
        self.features_name = []
        if self.args.create_new_data_for_classifier:
            self.create_data_bin()

            if self.args.load_data:
                self.c0l_create_proba_val = np.loadtxt(self.args.path_to_test + "c0l_create_proba_val.txt").astype('int32')[:100]
                self.c0r_create_proba_val = np.loadtxt(self.args.path_to_test + "c0r_create_proba_val.txt").astype('int32')[:100]
                self.c1l_create_proba_val = np.loadtxt(self.args.path_to_test + "c1l_create_proba_val.txt").astype('int32')[:100]
                self.c1r_create_proba_val = np.loadtxt(self.args.path_to_test + "c1r_create_proba_val.txt").astype('int32')[:100]
                self.Y_create_proba_val = np.loadtxt(self.args.path_to_test +"Y_create_proba_val.txt").astype('int32')[:100]


        else:

            self.c0l_create_proba_train = nn_model_ref.c0l_train_nn
            self.c0r_create_proba_train = nn_model_ref.c0r_train_nn
            self.c1l_create_proba_train = nn_model_ref.c1l_train_nn
            self.c1r_create_proba_train = nn_model_ref.c1r_train_nn
            self.Y_create_proba_train = nn_model_ref.Y_train_nn_binaire

            self.c0l_create_proba_val = nn_model_ref.c0l_val_nn
            self.c0r_create_proba_val = nn_model_ref.c0r_val_nn
            self.c1l_create_proba_val = nn_model_ref.c1l_val_nn
            self.c1r_create_proba_val = nn_model_ref.c1r_val_nn
            self.Y_create_proba_val = nn_model_ref.Y_val_nn_binaire

            self.X_bin_train = nn_model_ref.X_train_nn_binaire
            self.X_bin_val =nn_model_ref.X_val_nn_binaire
            self.Y_create_proba_train = nn_model_ref.Y_train_nn_binaire
            self.Y_create_proba_val = nn_model_ref.Y_val_nn_binaire


    def create_data_bin(self):
        self.X_bin_train, self.Y_create_proba_train, self.c0l_create_proba_train, self.c0r_create_proba_train, self.c1l_create_proba_train, self.c1r_create_proba_train = self.creator_data_binary.make_data(
            self.args.nbre_sample_train_classifier);
        self.X_bin_val, self.Y_create_proba_val, self.c0l_create_proba_val, self.c0r_create_proba_val, self.c1l_create_proba_val, self.c1r_create_proba_val = self.creator_data_binary.make_data(
            self.args.nbre_sample_val_classifier);

    def create_data_bin_val(self):
        self.X_bin_val, self.Y_create_proba_val, self.c0l_create_proba_val, self.c0r_create_proba_val, self.c1l_create_proba_val, self.c1r_create_proba_val = self.creator_data_binary.make_data(
            self.args.nbre_sample_val_classifier);

    def create_masked_moment(self, moment):
        masks_du_moment = []
        name_input_cic = ""
        for index_mask_all in range(len(self.args.inputs_type)):
            masks_du_moment.append(self.masks[index_mask_all][moment])
            name_input_cic += str(self.masks[index_mask_all][moment])
            name_input_cic += "_"
        name_input_cic = name_input_cic[:-1]
        self.features_name.append(name_input_cic)

        return masks_du_moment, name_input_cic

    def create_masked_inputs(self, liste_inputs, masks_du_moment):
        liste_inputsmasked = []
        for index, input_v in enumerate(liste_inputs):
            liste_inputsmasked.append((input_v) & masks_du_moment[index])
        debut = len(liste_inputs) * self.args.word_size - self.args.word_size
        ddt_entree = (np.uint64(liste_inputsmasked[0]) << debut)
        for index in range(1, len(liste_inputs)):
            ddt_entree += (np.uint64(liste_inputsmasked[index]) << debut - 16 * index)
        return ddt_entree


    def create_data(self,ToT, c0l, c0r, c1l, c1r, phase = "TRAIN"):
        num_samples = len(c0l)
        print()
        print("NUMBER OF SAMPLES FOR "+str(phase) + " :", num_samples)
        print()
        X_t = np.zeros((len(self.masks[0]), (len(c0l))), dtype=np.float16)
        liste_inputs = self.creator_data_binary.convert_data_inputs(self.args, c0l, c0r, c1l, c1r)
        for moment, _ in enumerate(tqdm(self.masks[0])):

            start = time.time()
            masks_du_moment, name_input_cic = self.create_masked_moment(moment)
            #print()
            #print(time.time() - start)
            ToT_du_moment = ToT[name_input_cic]
            #print(time.time() - start)
            ToT_entree = self.create_masked_inputs(liste_inputs, masks_du_moment)
            #print(time.time() - start)
            proba = np.zeros((len(ToT_entree),))
            #proba = ToT_du_moment.todok()[ToT_entree].toarray().squeeze()
            for index_v, v in enumerate(ToT_entree):
                try:
                    proba[index_v] = ToT_du_moment[v]
                except:
                    proba[index_v] = 0
            #print(proba)
            #print(time.time() - start)
            #print("--"*100)
            #proba = ToT_du_moment[ToT_entree]
            X_t[moment] = proba
        return X_t.transpose()

    def create_data_g(self,table_of_truth):
        ToT = table_of_truth.ToT
        self.X_proba_train = self.create_data(ToT, self.c0l_create_proba_train, self.c0r_create_proba_train, self.c1l_create_proba_train, self.c1r_create_proba_train, "TRAIN")
        self.X_proba_val =self.create_data(ToT, self.c0l_create_proba_val, self.c0r_create_proba_val, self.c1l_create_proba_val, self.c1r_create_proba_val, "VAL")
        print()

    def create_data_g_val(self,table_of_truth):
        ToT = table_of_truth.ToT
        self.X_proba_val =self.create_data(ToT, self.c0l_create_proba_val, self.c0r_create_proba_val, self.c1l_create_proba_val, self.c1r_create_proba_val, "VAL")
        print()
