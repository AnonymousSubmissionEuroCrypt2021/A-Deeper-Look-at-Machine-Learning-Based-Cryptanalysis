import numpy as np
from tqdm import tqdm
from scipy import sparse
#from sparse_vector import SparseVector


class ToT:

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
        self.mask_infos_compression = []
        self.mask_infos_hamming = []
        if self.args.create_new_data_for_ToT:
            self.create_data()
        else:
            self.c0l_create_ToT = nn_model_ref.c0l_train_nn
            self.c0r_create_ToT = nn_model_ref.c0r_train_nn
            self.c1l_create_ToT = nn_model_ref.c1l_train_nn
            self.c1r_create_ToT = nn_model_ref.c1r_train_nn
            self.Y_create_ToT = nn_model_ref.Y_train_nn_binaire
        if self.args.create_ToT_with_only_sample_from_cipher:
            self.c0l_create_ToT = self.c0l_create_ToT[self.Y_create_ToT == 1]
            self.c0r_create_ToT = self.c0r_create_ToT[self.Y_create_ToT == 1]
            self.c1l_create_ToT = self.c1l_create_ToT[self.Y_create_ToT == 1]
            self.c1r_create_ToT = self.c1r_create_ToT[self.Y_create_ToT == 1]

    def create_data(self):
        _, self.Y_create_ToT, self.c0l_create_ToT, self.c0r_create_ToT, self.c1l_create_ToT, self.c1r_create_ToT = self.creator_data_binary.make_data(
            self.args.nbre_sample_create_ToT);

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

        """   def convert_proba_pure(self, vals, counts, num_samples):

        nbre_param = len(vals)
        self.ms = int(np.log2(nbre_param))+1
        p_input_sachant_masks = counts / num_samples
        p_input_sachant_random = min(p_input_sachant_masks)
        assert min(p_input_sachant_masks) > 0
        assert max(p_input_sachant_masks) < 1
        a = p_input_sachant_random*0.99
        b = p_input_sachant_masks
        p_speck_sachant_input_masks_bar = self.mat_div2(a, b)
        assert min(p_speck_sachant_input_masks_bar) > 0
        assert max(p_speck_sachant_input_masks_bar) < 1
        p_speck_sachant_input_masks = 1- p_speck_sachant_input_masks_bar
        assert min(p_speck_sachant_input_masks) > 0
        assert max(p_speck_sachant_input_masks) < 1
        return p_speck_sachant_input_masks
        """

    def convert_proba_pure(self, vals, counts, num_samples):
        nbre_param = len(vals)
        self.ms = int(np.log2(nbre_param))
        p_input_sachant_masks = counts / num_samples
        # p_input_sachant_random = min(p_input_sachant_masks)
        assert min(p_input_sachant_masks) >= 0
        assert max(p_input_sachant_masks) <= 1
        numerateur = p_input_sachant_masks *0.5
        denominateur = p_input_sachant_masks * 0.5 + 0.5 * 2 ** (-self.ms)
        p_speck_sachant_input_masks_bar = self.mat_div(numerateur, denominateur)
        #print(min(p_speck_sachant_input_masks_bar), max(p_speck_sachant_input_masks_bar), self.ms)
        assert min(p_speck_sachant_input_masks_bar) > 0
        assert max(p_speck_sachant_input_masks_bar) < 1
        p_speck_sachant_input_masks = 1 - p_speck_sachant_input_masks_bar
        assert min(p_speck_sachant_input_masks) > 0
        assert max(p_speck_sachant_input_masks) < 1
        return p_speck_sachant_input_masks

    def convert_proba_impure(self, vals, counts, num_samples):
        nbre_param = len(vals)
        ms = int(np.log2(nbre_param))
        cste = 1 / (2 ** (ms))
        p_input_sachant_speK_masks = counts / num_samples
        a = 100 * (0.5 * p_input_sachant_speK_masks)
        b = cste + p_input_sachant_speK_masks
        p_speck_sachant_input_masks = self.mat_div(a, b)
        return p_speck_sachant_input_masks

    def create_DDT(self):
        num_samples = len(self.c0l_create_ToT)
        print("NUMBER OF SAMPLES IN DDT :", num_samples)
        print()
        self.nbre_param_ddt = 0
        liste_inputs = self.creator_data_binary.convert_data_inputs(self.args, self.c0l_create_ToT, self.c0r_create_ToT, self.c1l_create_ToT, self.c1r_create_ToT)
        self.ToT = {}
        #self.X_train_proba_train = np.zeros((len(self.masks[0]), (len(self.c0l_create_ToT))), dtype=np.float16)
        for moment, _ in enumerate(tqdm(self.masks[0])):
            masks_du_moment, name_input_cic = self.create_masked_moment(moment)
            hamming_number = np.sum([bin(x).count("1") for x in masks_du_moment])
            self.hamming_number = hamming_number
            ddt_entree = self.create_masked_inputs(liste_inputs, masks_du_moment)
            vals, counts = np.unique(ddt_entree, return_counts=True)
            self.nbre_param_ddt += len(vals)
            if self.args.create_ToT_with_only_sample_from_cipher:
                p_speck_sachant_input_masks = self.convert_proba_pure( vals, counts, num_samples)
            else:
                p_speck_sachant_input_masks = self.convert_proba_impure( vals, counts, num_samples)
            sv = dict(zip(vals, p_speck_sachant_input_masks))
            self.ToT[name_input_cic] = sv

            self.mask_infos_compression.append(1 - self.ms/hamming_number)
            self.mask_infos_hamming.append(hamming_number)

        print()
        print("NUMBER OF ENTRIES IN DDT :", self.nbre_param_ddt)
        print()


    def mat_div(self, a, b):
        return np.array([ra / rb  for ra, rb in zip(a, b)])

    def mat_div2(self, a, b):
        return np.array([a / rb  for rb in  b])