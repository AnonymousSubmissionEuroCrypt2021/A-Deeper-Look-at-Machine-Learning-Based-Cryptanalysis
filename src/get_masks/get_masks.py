import time
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from src.nn.DataLoader import DataLoader_cipher_binary
from captum.attr import Saliency, ShapleyValueSampling
from captum.attr import IntegratedGradients, DeepLift, GradientShap, NoiseTunnel, FeatureAblation, Occlusion
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm


class Get_masks:


    def __init__(self, args, net, path_file_models, rng, creator_data_binary, device):
        self.args = args
        self.t = Variable(torch.Tensor([0.5]))
        self.device= device
        self.net = net
        self.all_masks = []
        self.masks = [[] for x in range(len(self.args.inputs_type))]
        self.masks_infos = []
        self.path_file_models = path_file_models
        self.rng = rng
        self.creator_data_binary = creator_data_binary
        self.liste_max_min2 = [tuple(map(float, sub.split(', '))) for sub in args.liste_segmentation_prediction]
        if self.args.load_masks:
            self.load_masks()
            self.security_check()

    def save_masks(self, path_save_model):
        with open(path_save_model + "masks_all_v1.txt", "w") as file:
            for i in range(len(self.args.inputs_type)):
                file.write(str(self.masks[i]))
                file.write("\n")


    def start_step(self):
        self.create_data()
        self.cpt = 0
        for data_t, data_v in tqdm(zip(self.dataloader_train, self.dataloader_val)):
            if self.cpt < self.args.nbre_max_batch:
                self.cpt+=1
                interet_1, out_net_val, inputs_t, inputs_v, labels_v = self.eval_data(data_t, data_v)
                for (valmax, valimin) in self.liste_max_min2:
                    self.valmax_mnt = valmax
                    self.valimin_mnt = valimin
                    #print("Start segment: ", (valmax, valimin), " --- Nbre de masks: ", len(self.masks[0]))
                    #print()
                    dico_res_dico = {}
                    data_X_bin, data_Y, nbre_iter_max = self.get_data_interest_segmentation(valmax, valimin, out_net_val, interet_1, inputs_v, labels_v)
                    dico_res_dico["binary"] = data_X_bin
                    for methode_extraction in self.args.liste_methode_extraction:
                        dico_res_dico[methode_extraction] = data_X_bin.detach().cpu().numpy().copy()
                    for nbre_iter_ici in range(nbre_iter_max+1):
                        offeset = nbre_iter_ici*self.args.nbre_necessaire_val_SV
                        index_interet_v = data_X_bin[offeset: offeset + self.args.nbre_necessaire_val_SV]
                        if index_interet_v.shape[0] >0:
                            dico_res_dico = self.update_Xmasks(index_interet_v, dico_res_dico, offeset, inputs_t)

                    self.apply_selection(dico_res_dico)
        print()
        print("--- Number of masks: ", len(self.masks[0]))
        print()



    def extract_DL(self, X_test):
        dl = DeepLift(self.net)
        start = time.time()
        dl_attr_test = dl.attribute(X_test.to(self.device))
        #print("temps train", time.time() - start)
        return dl_attr_test.detach().cpu().numpy()

    def extract_IG(self, X_test, steps=50):
        ig = IntegratedGradients(self.net)
        start = time.time()
        ig_attr_test = ig.attribute(X_test.to(self.device), n_steps=steps)
        #print("temps train", time.time() - start)
        return ig_attr_test.detach().cpu().numpy()



    def create_data(self):
        self.X_deltaout_train, self.Y_tf, _, _, _, _= self.creator_data_binary.make_data(
            self.args.nbre_generate_data_train_val);
        self.X_eval, self.Y_eval, _, _, _, _ = self.creator_data_binary.make_data(
            self.args.nbre_generate_data_train_val);
        data_train = DataLoader_cipher_binary(self.X_deltaout_train, self.Y_tf, self.device)
        self.dataloader_train = DataLoader(data_train, batch_size=self.args.batch_size,
                                      shuffle=False, num_workers=self.args.num_workers)
        data_val = DataLoader_cipher_binary(self.X_eval, self.Y_eval, self.device)
        self.dataloader_val = DataLoader(data_val, batch_size=self.args.batch_size,
                                    shuffle=False, num_workers=self.args.num_workers)


    def eval_data(self, data_t, data_v):
        inputs_t, labels_t = data_t
        inputs_v, labels_v = data_v
        # out_net_train = self.net(inputs_t).squeeze(1)
        out_net_val = self.net(inputs_v.to(self.device))
        preds_val = (out_net_val.squeeze(1) > self.t.to(self.device)).float().cpu() * 1
        TP_val = (preds_val.eq(1) & labels_v.eq(1)).cpu()
        TN_val = (preds_val.eq(0) & labels_v.eq(0)).cpu()
        interet_1 = (TP_val | TN_val)
        return interet_1, out_net_val, inputs_t, inputs_v, labels_v




    def get_data_interest_segmentation(self, valmax, valimin, out_net_val, interet_1, inputs_v, labels_v):
        self.t2 = Variable(torch.Tensor([valimin]))
        self.t3 = Variable(torch.Tensor([valmax]))
        preds2_val1 = (self.t3.to(self.device) >= out_net_val.squeeze(1)).float().cpu() * 1
        preds2_val2 = (out_net_val.squeeze(1) >= self.t2.to(self.device)).float().cpu() * 1
        preds2_val = (preds2_val1.eq(1) & preds2_val2.eq(1)).cpu()
        interet = preds2_val & interet_1
        nbre_interet = interet.detach().cpu().numpy().sum(0)
        nbre_iter_max = nbre_interet // self.args.nbre_necessaire_val_SV
        data_X_bin = inputs_v[interet]
        data_Y = labels_v[interet]
        return data_X_bin, data_Y, nbre_iter_max


    def update_Xmasks(self, index_interet_v, dico_res_dico, offeset, inputs_t):
        #print(index_interet_v.shape)
        for methode_extraction in self.args.liste_methode_extraction:
            if methode_extraction == "IntegratedGradients":
                res = self.extract_IG(index_interet_v)
            if methode_extraction == "IntegratedGradients_tunnel":
                res = self.extract_IGNT(index_interet_v)
            if methode_extraction == "DeepLift":
                res = self.extract_DL(index_interet_v)
            if methode_extraction == "GradientShap":
                res = self.extract_GS(inputs_t, index_interet_v)
            if methode_extraction == "FeatureAblation":
                res = self.extract_FA(index_interet_v)
            if methode_extraction == "Saliency":
                res = self.extract_Sa(index_interet_v)
            if methode_extraction == "ShapleyValueSampling":
                res = self.extract_SV(index_interet_v)
            if methode_extraction == "Occlusion":
                res = self.extract_Oc(index_interet_v)
            dico_res_dico[methode_extraction][offeset: offeset + self.args.nbre_necessaire_val_SV] = res
        return dico_res_dico





    def extract_IGNT(self, X_test):
        ig = IntegratedGradients(self.net)
        ig_nt = NoiseTunnel(ig)
        start = time.time()
        ig_nt_attr_test = ig_nt.attribute(X_test.to(self.device))
        #print("temps train", time.time() - start)
        return ig_nt_attr_test.detach().cpu().numpy()


    def extract_GS(self, X_train, X_test):
        gs = GradientShap(self.net)
        start = time.time()
        gs_attr_test = gs.attribute(X_test.to(self.device), X_train.to(self.device))
        #print("temps train", time.time() - start)
        return gs_attr_test.detach().cpu().numpy()


    def extract_FA(self, X_test):
        fa = FeatureAblation(self.net)
        start = time.time()
        fa_attr_test = fa.attribute(X_test.to(self.device))
        #print("temps train", time.time() - start)
        return fa_attr_test.detach().cpu().numpy()

    def extract_Sa(self, X_test):
        saliency = Saliency(self.net)
        start = time.time()
        saliency_attr_test = saliency.attribute(X_test.to(self.device))
        #print("temps train", time.time() - start)
        return saliency_attr_test.detach().cpu().numpy()

    def extract_SV(self, X_test):
        Sv = ShapleyValueSampling(self.net)
        start = time.time()
        sv_attr_test = Sv.attribute(X_test.to(self.device))
        #print("temps train", time.time() - start)
        return sv_attr_test.detach().cpu().numpy()


    def extract_Oc(self, X_test, sliding_window = (4,)):
        oc = Occlusion(self.net)
        start = time.time()
        oc_attr_test = oc.attribute(X_test.to(self.device), sliding_window_shapes=sliding_window)
        #print("temps train", time.time() - start)
        return oc_attr_test.detach().cpu().numpy()

    def apply_selection(self, dico_res_dico):
        for_plot_dico = {}
        for key in list(dico_res_dico.keys()):
            if key == "binary":
                if "PCA" in self.args.liste_methode_selection:
                    for_plot_dico["PCA"] = {}
                    data_X_bin2 = self.transform_X_data(dico_res_dico[key], "PCA")
                    for_plot_dico["PCA"][key] = data_X_bin2
                    self.transform_into_mask_hw(data_X_bin2, key, "PCA")
                    self.transform_into_mask_thr(data_X_bin2, key, "PCA")
            else:
                for methode_selection in self.args.liste_methode_selection:
                    if methode_selection not in list(for_plot_dico.keys()):
                        for_plot_dico[methode_selection] = {}
                    data_X_bin2 = self.transform_X_data(dico_res_dico[key], methode_selection)
                    for_plot_dico[methode_selection][key] = data_X_bin2
                    self.transform_into_mask_hw(data_X_bin2, key, methode_selection)
                    self.transform_into_mask_thr(data_X_bin2, key, methode_selection)
        if self.args.save_fig_plot_feature_before_mask:
            self.plot_compare_method(for_plot_dico)
        #print("Nbre de masks:", len(self.masks[0]))


    def transform_X_data(self, data_X_bin, methode_selection):
        if methode_selection == "PCA":
            data_X_bin2 = self.PCA_method(data_X_bin)
        if methode_selection == "sum":
            data_X_bin2 = data_X_bin.sum(0)
        if methode_selection == "abs_sum":
            data_X_bin2 = np.abs(data_X_bin).sum(0)
        if methode_selection == "median":
            data_X_bin2 = np.median(data_X_bin, axis=0)
        if methode_selection == "abs_median":
            data_X_bin2 = np.median(np.abs(data_X_bin), axis=0)
        if methode_selection == "mean":
            data_X_bin2 = data_X_bin.mean(0)
        if methode_selection == "abs_mean":
            data_X_bin2 = np.abs(data_X_bin).mean(0)
        data_X_bin3 = data_X_bin2 / np.linalg.norm(data_X_bin2, ord=1)
        return data_X_bin3

    def PCA_method(self, X_train):
        pca = PCA(n_components=3)
        _ = pca.fit(X_train).transform(X_train)
        all_v = abs(pca.components_[0]) * 100 * pca.explained_variance_ratio_[0]
        for kkkk in range(1, 3):
            all_v += abs(pca.components_[kkkk]) * 100 * pca.explained_variance_ratio_[kkkk]
        return all_v

    def transform_into_mask_hw(self, data_X_bin2, key, methode_selection):
        nbit = self.args.word_size * len(self.args.inputs_type)
        for thrh in self.args.hamming_weigth:
            mask = np.zeros(nbit)
            all_vmask = np.argsort(data_X_bin2)[::-1][:int(thrh)]
            mask[all_vmask] = 1
            masks_for_moment = []
            for index_m in range(len(self.args.inputs_type)):
                masks_for_moment.append(int("".join(str(int(x)) for x in mask[ self.args.word_size * index_m: self.args.word_size * index_m +  self.args.word_size]), 2))
            infos_mask = "Segment_" + str(self.valmax_mnt) +"|" + str(self.valimin_mnt) + "_extraction_" + str(key) + "_selection_" + str(methode_selection) + "_thr_HAMMING_value" + str(thrh)
            self.add_mask(masks_for_moment, infos_mask)



    def transform_into_mask_thr(self, data_X_bin2,key, methode_selection):
        nbit = self.args.word_size * len(self.args.inputs_type)
        for thrh in self.args.thr_value:
            mask = np.zeros(nbit)
            all_vmask = data_X_bin2 > thrh
            mask[all_vmask] = 1
            masks_for_moment = []
            for index_m in range(len(self.args.inputs_type)):
                masks_for_moment.append(int("".join(str(int(x)) for x in mask[ self.args.word_size * index_m: self.args.word_size * index_m +  self.args.word_size]), 2))
            infos_mask = "Segment_" + str(self.valmax_mnt) + "|" + str(self.valimin_mnt) + "_extraction_" + str(
                key) + "_selection_" + str(methode_selection) + "_thr_HAMMING_value" + str(thrh)
            self.add_mask(masks_for_moment, infos_mask)


    def load_masks(self):
        masks_load = []
        liste_all_mask_load = []
        infile = open(self.args.file_mask, 'r')
        for line in infile:
            if "array" not in line:
                line = line.replace("[", "")
                line = line.replace("]", "")
                liste_mask = [int(x) for x in line.split(",")]
                nmask2 = len(liste_mask)
                masks_load.append(liste_mask[:self.args.nbre_max_masks_load])
        infile.close()
        for idxm, m in enumerate(masks_load[0]):
            liste_ici = []
            for i in range(len(self.args.inputs_type)):
                liste_ici.append(masks_load[i][idxm])
            liste_all_mask_load.append(tuple(liste_ici))
        for index_m in range(len(masks_load[0])):
            liste_ici = []
            for i in range(len(self.args.inputs_type)):
                liste_ici.append(masks_load[i][index_m])
            self.add_mask(liste_ici, self.args.file_mask)


    def add_mask(self, mask_to_add, infos_mask):
        if tuple(mask_to_add) not in self.all_masks:
            self.all_masks.append(tuple(mask_to_add))
            for i in range(len(self.args.inputs_type)):
                self.masks[i].append(mask_to_add[i])
            self.masks_infos += [infos_mask]

    def security_check(self):
        for i in range(len(self.args.inputs_type) - 1):
            assert len(self.masks[i]) == len(self.masks[i + 1])
        assert len(self.masks_infos) == len(self.masks[0])


    def plot_compare_method(self, for_plot_dico):

        for index_m, method in enumerate(list(for_plot_dico.keys())):

            fig, axs = plt.subplots(1, 1, figsize=(25, 10), facecolor='w', edgecolor='k')
            fig.subplots_adjust(hspace=.5, wspace=.001)

            #axs = axs.ravel()


            x_axis_data = np.arange(self.args.word_size * len(self.args.inputs_type))
            width = 0.14
            #x_axis_data_labels = list(map(lambda idx: feature_names[idx], x_axis_data))

            legends = list(for_plot_dico[method].keys())

            axs.set_title('Comparing input feature importances across multiple algorithms and ' + method + ' for segment ' + str(self.valmax_mnt) + "|" + str(self.valimin_mnt))
            axs.set_ylabel('Attributions')

            FONT_SIZE = 16
            plt.rc('font', size=FONT_SIZE)  # fontsize of the text sizes
            plt.rc('axes', titlesize=FONT_SIZE)  # fontsize of the axes title
            plt.rc('axes', labelsize=FONT_SIZE)  # fontsize of the x and y labels
            plt.rc('legend', fontsize=FONT_SIZE - 4)  # fontsize of the legend

            colors = ["#eb5e7c", "#A90000", '#34b8e0', '#4260f5', '#49ba81', 'yellow', "purple", "orange"]
            nbre=list(range(len(legends)))
            for index, leg, col in zip(nbre, legends, colors[:len(legends)]):
                axs.bar(x_axis_data + index * width, for_plot_dico[method][leg], width, align='center', alpha=0.8, color=col)

            axs.autoscale_view()
            plt.tight_layout()

            axs.set_xticks(x_axis_data + 0.5)
            axs.set_xticklabels(x_axis_data)


            title = 'Comparing input feature importances across multiple algorithms and ' + method + ' for segment ' + str(
                self.valmax_mnt) + "|" + str(self.valimin_mnt) + "number " + str(self.cpt) + " " + str(index_m)
            plt.legend(legends, loc=1)
            plt.savefig(self.path_file_models + title + ".png")
















