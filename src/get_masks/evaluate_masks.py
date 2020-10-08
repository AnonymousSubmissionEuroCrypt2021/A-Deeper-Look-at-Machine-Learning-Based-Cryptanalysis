from sklearn import linear_model
from sklearn.metrics import accuracy_score
from utils import make_train_data, convert_binary_to_probability, tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.tri as tri
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from sklearn.feature_selection import chi2, f_classif


def evaluate_masks(args, rng, masks, ddt_partials_u, path_save_model):
    X_deltaout, Y_tf, ctdata0l, ctdata0r, ctdata1l, ctdata1r = make_train_data(args, args.nbre_sample,
                                                                                   args.nombre_round_eval, rng)
    X_deltaout_v, Y_vf, ctdata0l_v, ctdata0r_v, ctdata1l_v, ctdata1r_v = make_train_data(args,
                                                                                         args.nbre_sampleval,
                                                                                         args.nombre_round_eval,
                                                                                         rng)

    res_all = []

    all_masks = []

    dico_train={}
    dico_val={}


    for masks_act in range(len(masks[0])):
        ddt_partials = {}
        hamming_nbre = 0
        masks_uniaue = [[masks[i][masks_act]] for i in range(len(masks))]

        name_input_cic = ""
        for i in range(len(masks)):
            hamming_nbre += np.sum(np.array([int(x) for x in list('{0:0b}'.format(masks_uniaue[i][0]))]))
            name_input_cic += str(masks_uniaue[i][0])
            name_input_cic += "_"
        name_input_cic = name_input_cic[:-1]

        all_masks.append(name_input_cic)

        ddt_partials[name_input_cic] = ddt_partials_u[name_input_cic]
        nbre_param = len(ddt_partials[list(ddt_partials.keys())[0]].keys())

        X_deltaout, X_DDT, feature_name = convert_binary_to_probability(args, ctdata0l, ctdata0r, ctdata1l,
                                                                        ctdata1r, ddt_partials, masks_uniaue,
                                                                        flag=False)

        X_deltaout_v, X_DDT_v, feature_name_v = convert_binary_to_probability(args, ctdata0l_v, ctdata0r_v,
                                                                            ctdata1l_v, ctdata1r_v,
                                                                            ddt_partials, masks_uniaue, flag=True)


        dico_train[name_input_cic] = [X_DDT, feature_name]
        dico_val[name_input_cic] = [X_DDT_v, feature_name_v]

        param_best = {'alpha': 0.3922345684859479,
                      'average': False,
                      'l1_ratio': 0.5605798090010486,
                      'loss': 'hinge',
                      'penalty': 'elasticnet',
                      'tol': 0.01}
        clf = linear_model.SGDClassifier(**param_best, random_state=args.seed)
        clf.fit(X_DDT, Y_tf)
        y_pred = clf.predict(X_DDT_v)


        res_all.append(
            [hamming_nbre, 1 - int(np.log2(nbre_param) + 0.1) / hamming_nbre, accuracy_score(y_pred=y_pred, y_true=Y_vf),
             "QQ"])
        # save_logs(path_save_model + 'logs_linear_reg.txt', y_pred, Y_vf, clf, X_DDT, Y_tf)
        # plot_coefficients(clf, feature_name, path_save_model, name=path_save_model + "features_importances_Linear.png")
        # https://towardsdatascience.com/feature-selection-techniques-for-classification-and-python-tips-for-their-application-10c0ddd7918b
        del ddt_partials

    res_all = np.array(res_all)
    np.save(path_save_model + 'masks_quality.txt', res_all)


    data = res_all
    data2 = np.zeros_like(data, dtype="float")
    data2[:, 0] = np.nan_to_num(np.array([int(x) for x in data[:, 0]]))
    data2[:, 1] = np.nan_to_num(np.array([float(x) for x in data[:, 1]]))
    data2[:, 2] = np.nan_to_num(np.array([float(x) for x in data[:, 2]]))
    data2[:, 3] = np.nan_to_num(np.array([1 for x in data[:, 3]]))


    data4 = np.load(args.path_random)
    data3 = np.zeros_like(data4, dtype="float")
    data3[:, 0] = np.array([int(x) for x in data4[:, 0]])
    data3[:, 1] = np.array([1 - float(x) for x in data4[:, 1]])
    data3[:, 2] = np.array([float(x) for x in data4[:, 2]])
    data3[:, 3] = np.array([0 for x in data4[:, 3]])



    data_f = np.concatenate((data2, data3))

    data_pd = pd.DataFrame(data_f, columns=["hamming", "compression", "accuracy_alone", "label"])
    data_pd = data_pd.sort_values(by=['hamming', "compression"])



    fig = plt.figure(figsize=(40, 15))

    plt.subplot(1, 3, 1)
    x = data_pd.hamming.values[data_pd['label'] == 1]
    ys = data_pd.compression.values[data_pd['label'] == 1]
    colors = cm.rainbow(np.linspace(0, 1, len(ys)))
    plt.scatter(x, ys, color=colors[0])

    x = data_pd.hamming.values[data_pd['label'] == 0]
    ys = data_pd.compression.values[data_pd['label'] == 0]
    plt.scatter(x, ys, color=colors[-1])
    plt.legend(["SHAPLEY MASKS", "RANDOM MASKS"])
    plt.xlabel("Number of Hamming")
    plt.ylabel("Compression of the DDT by the mask")
    plt.title("Compressions with Number of Hamming")

    plt.subplot(1, 3, 2)
    x = data_pd.hamming.values[data_pd['label'] == 1]
    ys = data_pd.accuracy_alone[data_pd['label'] == 1]
    plt.scatter(x, ys, color=colors[0])
    x = data_pd.hamming.values[data_pd['label'] == 0]
    ys = data_pd.accuracy_alone.values[data_pd['label'] == 0]
    plt.scatter(x, ys, color=colors[-1])
    plt.legend(["SHAPLEY MASKS", "RANDOM MASKS"])
    plt.xlabel("Number of Hamming")
    plt.ylabel("Accuracy of the mask alone")
    plt.title("Accuracy with Number of Hamming")

    plt.subplot(1, 3, 3)
    x = data_pd.compression.values[data_pd['label'] == 1]
    ys = data_pd.accuracy_alone[data_pd['label'] == 1]
    plt.scatter(x, ys, color=colors[0])
    x = data_pd.compression.values[data_pd['label'] == 0]
    ys = data_pd.accuracy_alone.values[data_pd['label'] == 0]
    plt.scatter(x, ys, color=colors[-1])
    plt.legend(["SHAPLEY MASKS", "RANDOM MASKS"])
    plt.ylabel("Accuracy of the mask alone")
    plt.xlabel("Compression of the DDT by the mask")
    plt.title("Accuracy with compression")

    fig.suptitle('Plot of the 3 characteristic that make a good mask for SHAPLEY masks (blue) and random masks (red)',
                 fontsize=30)
    plt.savefig(path_save_model + "2D real plot.png")
    

    fig = plt.figure(figsize=(30, 30))
    ax = fig.add_subplot(111, projection='3d')

    # For each set of style and range settings, plot n random points in the box
    # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    for index, m, color_ici in [(1, 'o', 0), (0, '^', -1)]:
        xs = data_pd.hamming.values[data_pd['label'] == index]
        ys = data_pd.compression.values[data_pd['label'] == index]
        zs = data_pd.accuracy_alone.values[data_pd['label'] == index]
        ax.scatter(xs, ys, zs, marker=m, color=colors[color_ici], s=30)
    ax.legend(["SHAPLEY masks", "Random masks"])

    ax.set_xlabel('Number of Hamming')
    ax.set_ylabel('Compression of the DDT by the mask')
    ax.set_zlabel('Accuracy of the mask alone')
    fig.suptitle('Plot of the 3 characteristic that make a good mask for SHAPLEY masks (blue) and random masks (red)',
                 fontsize=16)
    plt.savefig(path_save_model + "3D real plot.png")
    



    def circleOfCorrelations(pc_infos, ebouli):
        plt.subplot(1, 2, 1)
        plt.Circle((10, 15), radius=1, color='g', fill=False)
        circle1 = plt.Circle((0, 0), radius=1, color='g', fill=False)
        fig = plt.gcf()
        fig.gca().add_artist(circle1)
        for idx in range(len(pc_infos["PC-0"])):
            x = pc_infos["PC-0"][idx]
            y = pc_infos["PC-1"][idx]
            plt.plot([0.0, x], [0.0, y], 'k-')
            plt.plot(x, y, 'rx')
            plt.annotate(pc_infos.index[idx], xy=(x, y))
        plt.xlabel("PC-1 (%s%%)" % str(ebouli[0])[:4].lstrip("0."))
        plt.ylabel("PC-2 (%s%%)" % str(ebouli[1])[:4].lstrip("0."))
        plt.xlim((-1, 1))
        plt.ylim((-1, 1))
        plt.axhline(y=0, color='k', ls='--')
        plt.axvline(x=0, c='k', ls='--')
        plt.title("Circle of Correlations")

    def circleOfCorrelations2(pc_infos, ebouli):
        plt.subplot(1, 2, 1)
        plt.Circle((10, 15), radius=1, color='g', fill=False)
        circle1 = plt.Circle((0, 0), radius=1, color='g', fill=False)
        fig = plt.gcf()
        fig.gca().add_artist(circle1)
        for idx in range(len(pc_infos["PC-0"])):
            x = pc_infos["PC-0"][idx]
            y = pc_infos["PC-2"][idx]
            plt.plot([0.0, x], [0.0, y], 'k-')
            plt.plot(x, y, 'rx')
            plt.annotate(pc_infos.index[idx], xy=(x, y))
        plt.xlabel("PC-1 (%s%%)" % str(ebouli[0])[:4].lstrip("0."))
        plt.ylabel("PC-2 (%s%%)" % str(ebouli[2])[:4].lstrip("0."))
        plt.xlim((-1, 1))
        plt.ylim((-1, 1))
        plt.axhline(y=0, color='k', ls='--')
        plt.axvline(x=0, c='k', ls='--')
        plt.title("Circle of Correlations")

    def myPCA(df, color_ici):
        # Normalize data
        plt.figure(figsize=(30, 15))
        df_norm = StandardScaler().fit_transform(df)  # (df - df.mean()) / df.std()
        # PCA
        df_norm = np.nan_to_num(df_norm)
        pca = PCA(n_components=3)  # n_components='mle')
        pca_res = pca.fit_transform(df_norm)  # pca.fit_transform(df_norm.values)
        # Ebouli
        ebouli = pd.Series(pca.explained_variance_ratio_)
        # Circle of correlations
        # http://stackoverflow.com/a/22996786/1565438
        coef = np.transpose(pca.components_)
        cols = ['PC-' + str(x) for x in range(len(ebouli))]
        pc_infos = pd.DataFrame(coef, columns=cols, index=df.columns)
        circleOfCorrelations(pc_infos, ebouli)
        plt.subplot(1, 2, 2)
        dat = pd.DataFrame(pca_res, columns=cols)
        # print(dat["PC-0"])
        # print(dat["PC-1"])
        plt.scatter(dat["PC-0"].values, dat["PC-1"].values, color=color_ici)
        plt.xlabel("PC-1 (%s%%)" % str(ebouli[0])[:4].lstrip("0."))
        plt.ylabel("PC-2 (%s%%)" % str(ebouli[1])[:4].lstrip("0."))
        plt.title("PCA")
        plt.savefig(path_save_model + str(color_ici) + "2D ACP plot pca1-2 .png")
        

    def myPCA2(df, label_QQ, labelrandom, color1, color2):
        # Normalize data
        plt.figure(figsize=(30, 15))
        df_norm = StandardScaler().fit_transform(df)  # (df - df.mean()) / df.std()
        # PCA
        df_norm = np.nan_to_num(df_norm)
        pca = PCA(n_components=3)  # n_components='mle')
        pca_res = pca.fit_transform(df_norm)  # pca.fit_transform(df_norm.values)
        # Ebouli
        ebouli = pd.Series(pca.explained_variance_ratio_)
        # Circle of correlations
        # http://stackoverflow.com/a/22996786/1565438
        coef = np.transpose(pca.components_)
        cols = ['PC-' + str(x) for x in range(len(ebouli))]
        pc_infos = pd.DataFrame(coef, columns=cols, index=df.columns)
        circleOfCorrelations(pc_infos, ebouli)
        plt.subplot(1, 2, 2)
        dat = pd.DataFrame(pca_res, columns=cols)
        # print(dat["PC-0"])
        # print(dat["PC-1"])
        plt.scatter(dat["PC-0"].values[label_QQ], dat["PC-1"].values[label_QQ], color=color1)
        plt.scatter(dat["PC-0"].values[labelrandom], dat["PC-1"].values[labelrandom], color=color2)
        plt.xlabel("PC-1 (%s%%)" % str(ebouli[0])[:4].lstrip("0."))
        plt.ylabel("PC-3 (%s%%)" % str(ebouli[1])[:4].lstrip("0."))
        plt.title("PCA")
        plt.savefig(path_save_model + str(color_ici) + "2D ACP plot comparaison pca1-2.png")

        
        return pca_res

    def myPCA_v2(df, color_ici):
        # Normalize data
        plt.figure(figsize=(30, 15))
        df_norm = StandardScaler().fit_transform(df)  # (df - df.mean()) / df.std()
        # PCA
        df_norm = np.nan_to_num(df_norm)
        pca = PCA(n_components=3)  # n_components='mle')
        pca_res = pca.fit_transform(df_norm)  # pca.fit_transform(df_norm.values)
        # Ebouli
        ebouli = pd.Series(pca.explained_variance_ratio_)
        # Circle of correlations
        # http://stackoverflow.com/a/22996786/1565438
        coef = np.transpose(pca.components_)
        cols = ['PC-' + str(x) for x in range(len(ebouli))]
        pc_infos = pd.DataFrame(coef, columns=cols, index=df.columns)
        circleOfCorrelations2(pc_infos, ebouli)
        plt.subplot(1, 2, 2)
        dat = pd.DataFrame(pca_res, columns=cols)
        # print(dat["PC-0"])
        # print(dat["PC-1"])
        plt.scatter(dat["PC-0"].values, dat["PC-2"].values, color=color_ici)
        plt.xlabel("PC-1 (%s%%)" % str(ebouli[0])[:4].lstrip("0."))
        plt.ylabel("PC-3 (%s%%)" % str(ebouli[2])[:4].lstrip("0."))
        plt.title("PCA")
        plt.savefig(path_save_model + str(color_ici) + "2D ACP plot pca 1 -3 .png")

        

    def myPCA2_v2(df, label_QQ, labelrandom, color1, color2):
        # Normalize data
        plt.figure(figsize=(30, 15))
        df_norm = StandardScaler().fit_transform(df)  # (df - df.mean()) / df.std()
        # PCA
        df_norm = np.nan_to_num(df_norm)
        pca = PCA(n_components=3)  # n_components='mle')
        pca_res = pca.fit_transform(df_norm)  # pca.fit_transform(df_norm.values)
        # Ebouli
        ebouli = pd.Series(pca.explained_variance_ratio_)
        # Circle of correlations
        # http://stackoverflow.com/a/22996786/1565438
        coef = np.transpose(pca.components_)
        cols = ['PC-' + str(x) for x in range(len(ebouli))]
        pc_infos = pd.DataFrame(coef, columns=cols, index=df.columns)
        circleOfCorrelations2(pc_infos, ebouli)
        plt.subplot(1, 2, 2)
        dat = pd.DataFrame(pca_res, columns=cols)
        # print(dat["PC-0"])
        # print(dat["PC-1"])
        plt.scatter(dat["PC-0"].values[label_QQ], dat["PC-2"].values[label_QQ], color=color1)
        plt.scatter(dat["PC-0"].values[labelrandom], dat["PC-2"].values[labelrandom], color=color2)
        plt.xlabel("PC-1 (%s%%)" % str(ebouli[0])[:4].lstrip("0."))
        plt.ylabel("PC-3 (%s%%)" % str(ebouli[2])[:4].lstrip("0."))
        plt.title("PCA")

        plt.savefig(path_save_model + str(color_ici) + "2D ACP plot comparaison pca1 - 3.png")

        

    #df = data_pd[data_pd['label'] == 1].drop("label", axis=1)
    #myPCA(df, colors[0])
    #myPCA_v2(df, colors[0])

    #df = data_pd[data_pd['label'] == 0].drop("label", axis=1)
    #myPCA(df, colors[-1])
    #myPCA_v2(df, colors[-1])

    #df = data_pd.drop("label", axis=1)
    #pca_res = myPCA2(df, data_pd['label'] == 1, data_pd['label'] == 0, colors[0], colors[-1])
    #myPCA2_v2(df, data_pd['label'] == 1, data_pd['label'] == 0, colors[0], colors[-1])

    max_compression = max(data_pd.compression.values) + 0.1
    max_accuracy = max(data_pd.accuracy_alone.values) + 0.1
    min_accuracy = min(data_pd.accuracy_alone.values) - 0.1

    npts = 400
    ngridx = 400
    ngridy = 400

    hamming = np.random.uniform(0, 1, npts)
    compression = np.random.uniform(0, max_compression, npts)
    accuracy = np.random.uniform(min_accuracy, max_accuracy, npts)

    _, score1, score2, score3 = get_score_masks(npts, hamming, compression, accuracy)



    hamming_i = np.linspace(-0.1, 1.1, ngridx)
    compression_i = np.linspace(-0.1, max_compression, ngridy)
    accuracy_i = np.linspace(min_accuracy, max_accuracy, ngridy)

    triang = tri.Triangulation(hamming, compression)
    interpolator = tri.LinearTriInterpolator(triang, score1)
    Xi, Yi = np.meshgrid(hamming_i, compression_i)
    zi = interpolator(Xi, Yi)

    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(40, 15))
    CS = ax1.contour(hamming_i, compression_i, zi, levels=14, linewidths=0.5, colors='k')
    cntr1 = ax1.contourf(hamming_i, compression_i, zi, levels=14, cmap="RdBu_r")
    fig.colorbar(cntr1, ax=ax1)
    ax1.clabel(CS, inline=1, fontsize=10)
    ax1.set(xlim=(0, 1), ylim=(0, max_compression))
    ax1.set_title('Metric Hamming vs Compression')
    ax1.set_xlabel('Hamming')
    ax1.set_ylabel('Compression')

    triang = tri.Triangulation(hamming, accuracy)
    interpolator = tri.LinearTriInterpolator(triang, score2)
    Xi, Yi = np.meshgrid(hamming_i, accuracy_i)
    zi = interpolator(Xi, Yi)

    CS = ax2.contour(hamming_i, accuracy_i, zi, levels=14, linewidths=0.5, colors='k')
    cntr1 = ax2.contourf(hamming_i, accuracy_i, zi, levels=14, cmap="RdBu_r")
    fig.colorbar(cntr1, ax=ax2)
    ax2.clabel(CS, inline=1, fontsize=10)
    ax2.set(xlim=(0, 1), ylim=(min_accuracy, max_accuracy))
    ax2.set_title('Metric Hamming vs accuracy')
    ax2.set_xlabel('Hamming')
    ax2.set_ylabel('accuracy')

    triang = tri.Triangulation(compression, accuracy)
    interpolator = tri.LinearTriInterpolator(triang, score3)
    Xi, Yi = np.meshgrid(compression_i, accuracy_i)
    zi = interpolator(Xi, Yi)

    CS = ax3.contour(compression_i, accuracy_i, zi, levels=14, linewidths=0.5, colors='k')
    cntr1 = ax3.contourf(compression_i, accuracy_i, zi, levels=14, cmap="RdBu_r")
    fig.colorbar(cntr1, ax=ax3)
    ax3.clabel(CS, inline=1, fontsize=10)
    ax3.set(xlim=(0, max_compression), ylim=(min_accuracy, max_accuracy))
    ax3.set_title('Metric Compression vs accuracy')
    ax3.set_xlabel('Compression')
    ax3.set_ylabel('accuracy')

    colors = cm.rainbow(np.linspace(0, 1, len(data_pd.compression.values)))
    ax1.plot(data_pd.hamming.values[data_pd['label'] == 1] / 32, data_pd.compression.values[data_pd['label'] == 1],
             'ko', ms=3, color=colors[0])
    ax2.plot(data_pd.hamming.values[data_pd['label'] == 1] / 32, data_pd.accuracy_alone.values[data_pd['label'] == 1],
             'ko', ms=3, color=colors[0])
    ax3.plot(data_pd.compression.values[data_pd['label'] == 1], data_pd.accuracy_alone.values[data_pd['label'] == 1],
             'ko', ms=3, color=colors[0])

    ax1.plot(data_pd.hamming.values[data_pd['label'] == 0] / 32, data_pd.compression.values[data_pd['label'] == 0],
             'ko', ms=3, color=colors[-1])
    plt.legend(["QQ MASKS", "RANDOM MASKS"])
    ax2.plot(data_pd.hamming.values[data_pd['label'] == 0] / 32, data_pd.accuracy_alone.values[data_pd['label'] == 0],
             'ko', ms=3, color=colors[-1])
    plt.legend(["QQ MASKS", "RANDOM MASKS"])
    ax3.plot(data_pd.compression.values[data_pd['label'] == 0], data_pd.accuracy_alone.values[data_pd['label'] == 0],
             'ko', ms=3, color=colors[-1])
    plt.legend(["QQ MASKS", "RANDOM MASKS"])
    plt.savefig(path_save_model + "metric.png")

    npts = len(data_pd.hamming.values[data_pd['label'] == 1] / 32)
    hamming = data_pd.hamming.values[data_pd['label'] == 1] / 32
    compression = data_pd.compression.values[data_pd['label'] == 1]
    accuracy = data_pd.accuracy_alone.values[data_pd['label'] == 1]

    score_f, score1, score2, score3 = get_score_masks(npts, hamming, compression, accuracy)

    print("")
    print("SCORE OD MASKS:")
    print(score_f)

    print("")
    d = {'masks': all_masks, 'score': score_f}

    df = pd.DataFrame(data=d)

    df = df.sort_values(by=['score'], ascending=False)


    df.to_csv(path_save_model + "masks_and_score.csv", index=False)

    masks_to_keep = df.masks.values

    if args.select_maks_thr_score:
        masks_to_keep = masks_to_keep[df['score'] > args.thr_score]

    if args.select_maks_max_num:
        masks_to_keep = masks_to_keep[:args.max_num]

    masks_final = [[] for x in range(len(args.inputs_type))]
    for mask_ici in masks_to_keep:
        list_mask_ici = mask_ici.split("_")
        for x_index, x_value in enumerate(list_mask_ici):
            masks_final[x_index].append(int(x_value))

    X_DDT_all = np.zeros((len(masks_to_keep), len(ctdata0l ^ ctdata1l)), dtype=np.float16)
    X_DDT_all_v = np.zeros((len(masks_to_keep), len(ctdata0l_v ^ ctdata1l_v)), dtype=np.float16)

    feature_name_all = []
    for mask_ici_index, mask_ici in enumerate(masks_to_keep):
        X_DDT_all[mask_ici_index] = np.squeeze(dico_train[mask_ici][0])
        X_DDT_all_v[mask_ici_index] = np.squeeze(dico_val[mask_ici][0])
        feature_name_all.append(dico_train[mask_ici][1][0])

        dico_train[mask_ici] = 0
        dico_val[mask_ici] = 0

    del dico_train, dico_val


    print("NUMBER OF MASKS FINAL :", len(masks_to_keep))

    dico_final = dict((k, ddt_partials_u[k]) for k in masks_to_keep if k in ddt_partials_u)

    competeur = 0
    for name_input_cic in dico_final.keys():
        competeur += len(dico_final[name_input_cic])

    print()
    print("Nmbre de parametres dans DDT:", competeur)
    print()




    X_DDT_df = pd.DataFrame(X_DDT_all.transpose(), columns=feature_name_all)
    Y_tf_df = pd.DataFrame(Y_tf)
    X_DDT_v_df = pd.DataFrame(X_DDT_all_v.transpose(), columns=feature_name_all)
    Y_vf_df = pd.DataFrame(Y_vf)


    X_DDT_df.to_csv(path_save_model + "X_DDT_df.csv", index=False)
    Y_tf_df.to_csv(path_save_model + "Y_tf_df.csv", index=False)
    X_DDT_v_df.to_csv(path_save_model + "X_DDT_v_df.csv", index=False)
    Y_vf_df.to_csv(path_save_model + "Y_vf_df.csv", index=False)

    index_interet_1 = Y_tf_df.values == 1
    df_1 = X_DDT_df[index_interet_1]

    index_interet_0 = Y_tf_df.values == 0
    df_0 = X_DDT_df[index_interet_0]

    print()
    print("START COMAPRASION HISTOGRAMM SAMPLES")
    print()


    plt.figure(figsize=(20, 7))
    for i, binwidth in enumerate([5]):
        # Set up the plot
        ax = plt.subplot(1, 2, 1)

        # Draw the plot
        ax.hist(df_1.sum(axis=1) / len(feature_name_all), bins=int(180 / binwidth),
                color='blue', edgecolor='black')

        # Title and labels
        ax.set_title('Histogram of SPECK', size=30)
        ax.set_xlabel('Probability', size=22)
        ax.set_ylabel('Number of samples', size=22)

        # Set up the plot
        ax = plt.subplot(1, 2, 2)

        # Draw the plot
        ax.hist(df_0.sum(axis=1) / len(feature_name_all), bins=int(180 / binwidth),
                color='blue', edgecolor='black')

        # Title and labels
        ax.set_title('Histogram of RANDOM', size=30)
        ax.set_xlabel('Probability', size=22)
        ax.set_ylabel('Number of samples', size=22)

    plt.tight_layout()
    plt.savefig(path_save_model + "histogramm.png")

    print()
    print("START COMAPRASION HISTOGRAMM FEATURS")
    print()

    plt.figure(figsize=(25, 15))



    df_11 = df_1 / 100.0
    df_00 = df_0 / 100.0

    x1 = 100*df_11.sum().astype(np.float64) / len(df_1)
    x2 = 100*df_00.sum().astype(np.float64) / len(df_0)


    # Assign colors for each airline and the names
    colors = ['#E69F00', '#56B4E9']
    names = ['SPECK', 'RANDOM']

    # Make the histogram using a list of lists
    # Normalize the flights and assign colors and names
    indices = [k for k in range(len(feature_name_all))]
    # Calculate optimal width
    width = np.min(np.diff(indices)) / 3

    # Make the histogram using a list of lists
    # Normalize the flights and assign colors and names
    plt.bar(indices - width, x1, width, color='b', label='SPECK')
    plt.bar(indices, x2, width, color='r', label='RANDOM')

    # Plot formatting
    plt.legend()
    plt.xlabel(' FEATURE NUMBER ')
    plt.ylabel('Normalized SUM 1 ')
    plt.title('COMPARASION FEATURES IMPORANTANCES')

    plt.savefig(path_save_model + "COMPARASION FEATURES IMPORANTANCES.png")

    print()
    print("START INDEPENACE TEST FEATURES LABELS")
    print()

    df = X_DDT_df
    X = df
    y = Y_tf_df
    chi_scores = f_classif(X, y)
    plt.figure(figsize=(25, 15))

    p_values = pd.Series(chi_scores[1], index=X.columns)
    p_values.sort_values(ascending=False, inplace=True)

    p_values.to_csv(path_save_model + "START INDEPENACE TEST FEATURES LABELS.csv")


    ax = p_values.plot.bar()
    fig = ax.get_figure()
    fig.savefig(path_save_model + "like_variables_results.png")

    print()
    print("START INDEPENACE TEST FEATURES FEATURES")
    print()

    alpha = 0.05
    res = np.zeros((len(feature_name_all), len(feature_name_all)))
    for i, _ in enumerate(tqdm(feature_name_all)):
        if i < len(feature_name_all) - 1:
            feature_name_ici = str(feature_name_all[i])
            X = df.drop(feature_name_ici, axis=1)
            y = df[feature_name_ici]
            chi_scores = f_classif(X, y)
            p_values = pd.Series(chi_scores[1], index=X.columns)
            p_values.sort_values(ascending=False, inplace=True)
            for index_index, index_v in enumerate(p_values.index):
                index_v_new = feature_name_all.index(index_v)
                res[i, int(index_v_new)] = p_values.values[index_index]
            del X, y
            if len(df.columns) > 1:
                df = df.drop(feature_name_ici, axis=1)

    df = pd.DataFrame(res, index=feature_name_all, columns=feature_name_all)
    vals = np.around(df.values, 2)
    norm = plt.Normalize(vals.min() - 1, vals.max() + 1)
    colours = plt.cm.RdBu(vals)

    fig = plt.figure(figsize=(100, 100))
    ax = fig.add_subplot(111, frameon=True, xticks=[], yticks=[])

    the_table = plt.table(cellText=vals, rowLabels=df.index, colLabels=df.columns,
                          colWidths=[0.03] * vals.shape[1], loc='center',
                          cellColours=colours)

    plt.savefig(path_save_model + "COMPARASION INTRA FEATURES XI 2.png")
    df.to_csv(path_save_model + "COMPARASION INTRA FEATURES XI 2.csv", index=False)


    return dico_final, masks_final, X_DDT_all.transpose(), X_DDT_all_v.transpose(), feature_name_all, Y_tf, Y_vf






def get_score_masks(npts, hamming, compression, accuracy):
    score1 = np.random.uniform(0, 1, npts)
    score2 = np.random.uniform(0, 1, npts)
    score3 = np.random.uniform(0, 1, npts)

    for i in range(len(hamming)):
        if compression[i] ==0:
            score1[i] = 0
        else:
            score1[i] = (1-np.sqrt(0.25*(1-hamming[i])**2 + (1-compression[i])**2))

    for i in range(len(hamming)):
        if accuracy[i] <0.5:
            score2[i] = 0
        else:
            score2[i] = 1

    for i in range(len(compression)):
        if accuracy[i]>0.6 and compression[i] > 0.1:
            score3[i] = (accuracy[i] - 0.6) + (compression[i] - 0.1)
        if accuracy[i] < 0.6 and compression[i] < 0.1:
            score3[i] = (accuracy[i] - 0.6) + (compression[i] - 0.1)
        if accuracy[i] > 0.6 and compression[i] < 0.1:
            score3[i] = (accuracy[i] - 0.6) * (compression[i] - 0.1)
        if accuracy[i] < 0.6 and compression[i] > 0.1:
            score3[i] = -1 * (accuracy[i] - 0.6) * (1.5*compression[i] - 0.1)


    score_f = score2*(score1 + score3)

    return 100*score_f, score1, score2, score3
