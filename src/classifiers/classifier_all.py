from src.classifiers.nn_classifier_keras import train_speck_distinguisher, train_speck_distinguisher2
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.tree import export_graphviz
import os
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.svm import SVC

class All_classifier:


    def __init__(self, args, path_save_model, generator_data, get_masks_gen, nn_model_ref, table_of_truth, cpt = 0):
        self.args = args
        self.cpt = cpt
        self.table_of_truth = table_of_truth
        self.path_save_model = path_save_model
        self.nn_model_ref = nn_model_ref
        self.get_masks_gen = get_masks_gen
        self.generator_data = generator_data
        self.X_train_proba = generator_data.X_proba_train
        self.Y_train_proba = generator_data.Y_create_proba_train
        self.X_eval_proba = generator_data.X_proba_val
        self.Y_eval_proba = generator_data.Y_create_proba_val
        X_t_df = pd.DataFrame(self.X_train_proba, columns=self.table_of_truth.features_name)
        X_v_df = pd.DataFrame(self.X_eval_proba, columns=self.table_of_truth.features_name)
        Y_t_df = pd.DataFrame(self.Y_train_proba, columns=["Label"])
        Y_v_df = pd.DataFrame(self.Y_eval_proba, columns=["Label"])
        if self.args.save_data_proba:
            print("START SAVE DATA PROBA")
            #X_t_df.to_csv(path_save_model + "X_train_proba.csv", index=False)
            #X_v_df.to_csv(path_save_model + "X_val_proba.csv", index=False)
            Y_t_df.to_csv(path_save_model + str(cpt) + "_Y_train_proba.csv", index=False)
            Y_v_df.to_csv(path_save_model + str(cpt) + "_Y_val_proba.csv", index=False)
            print("END SAVE DATA PROBA")
        self.masks_infos_score = None
        self.masks_infos_rank = None
        self.clf_final = {}
        if args.retrain_nn_ref:
            net_retrain = self.retrain_classifier_final(args, nn_model_ref)
            self.clf_final["NN_ref_retrain"] = net_retrain



    def classify_all(self):
        for clf in self.args.classifiers_ours:
            if clf == "NN":
                print("START CLASSIFY NN")
                print()
                classifier = self.classifier_nn()
                self.clf_final["NN"] = classifier
            if clf == "LGBM":
                print("START CLASSIFY LGBM")
                print()
                classifier = self.classifier_lgbm()
                self.clf_final["LGBM"] = classifier
                if self.args.retrain_with_import_features and self.args.keep_number_most_impactfull >0:
                    classifier, indices = self.classifier_lgbm_retrict()
                    self.clf_final["LGBM_restricted"] = [classifier, indices]
            if clf == "RF":
                print("START CLASSIFY RF")
                print()
                classifier = self.classifier_RF()
                self.clf_final["RF"] = classifier
                if self.args.retrain_with_import_features and self.args.keep_number_most_impactfull >0:
                    classifier, indices = self.classifier_RF_retrict()
                    self.clf_final["RF_restricted"] = [classifier, indices]
            if clf == "LR":
                print("START CLASSIFY LR")
                print()
                classifier = self.classifier_lr()
                self.clf_final["LR"] = classifier
                #if self.args.retrain_with_import_features and self.args.keep_number_most_impactfull >0:
                #    classifier, indices = self.classifier_lgbm_retrict()
                #    self.clf_final["LR_restricted"] = [classifier, indices]




    def retrain_classifier_final(self, args, nn_model_ref):
        nn_model_ref.epochs = args.num_epch_2
        nn_model_ref.batch_size_2 = args.batch_size_2
        nn_model_ref.net.freeze()
        X_train_proba_feat, X_eval_proba_feat = nn_model_ref.all_intermediaire, nn_model_ref.all_intermediaire_val
        Y_train_proba = nn_model_ref.Y_train_nn_binaire
        Y_eval_proba =  nn_model_ref.Y_val_nn_binaire
        print("START RETRAIN LINEAR NN GOHR ")
        print()
        """net_retrain, h = train_speck_distinguisher(args, X_train_proba_feat.shape[1], X_train_proba_feat,
                                                   Y_train_proba, X_eval_proba_feat, Y_eval_proba,
                                                   bs=args.batch_size_2,
                                                   epoch=args.num_epch_2, name_ici="retrain_nn_gohr",
                                                   wdir=self.path_save_model)"""



        from alibi.explainers import AnchorTabular
        #from alibi.explainers import AnchorImage
        from sklearn.ensemble import RandomForestClassifier

        clf = RandomForestClassifier(n_estimators=50)
        clf.fit(X_train_proba_feat, Y_train_proba)
        predict_fn = lambda x: clf.predict_proba(x)
        feature_names = [i for i in range(X_train_proba_feat.shape[1])]
        explainer = AnchorTabular(predict_fn, feature_names)
        idx = 0
        explainer.fit(X_train_proba_feat, disc_perc=(25))
        print('Prediction: ', explainer.predictor(X_eval_proba_feat[idx].reshape(1, -1))[0])

        #print('Prediction: ', explainer.predict_fn(X_eval_proba_feat[idx].reshape(1, -1))[0])
        explanation = explainer.explain(X_eval_proba_feat[idx], threshold=0.8)
        print('Anchor: %s' % (' AND '.join(explanation['names'])))
        print('Precision: %.2f' % explanation['precision'])
        print('Coverage: %.2f' % explanation['coverage'])

        print(ok)






        return net_retrain


    def classifier_nn(self):
        net2, h = train_speck_distinguisher2(self.args, self.X_train_proba.shape[1], self.X_train_proba,
                                            self.Y_train_proba, self.X_eval_proba, self.Y_eval_proba,
                                            bs=self.args.batch_size_our,
                                            epoch=self.args.num_epch_our, name_ici="our_model",
                                            wdir=self.path_save_model, flag_3layes=False)
        return net2


    def classifier_lgbm_general(self, X_DDTpd, X_eval, features):
        best_params_ = {
            'objective': 'binary',
            'num_leaves': 50,
            'min_data_in_leaf': 10,
            'max_depth': 10,
            'max_bin': 50,
            'learning_rate': 0.01,
            'dart': False,
            'reg_alpha': 0.1,
            'reg_lambda': 0,
            'n_estimators': 1000,
            'bootstrap': True,
            'dart': False
        }

        scaler = StandardScaler().fit(X_DDTpd)
        X_DDTpd = scaler.transform(X_DDTpd)
        X_eval = scaler.transform(X_eval)
        #poly = PolynomialFeatures(2)
        #X_DDTpd = poly.fit_transform(X_DDTpd)
        #X_eval = poly.fit_transform
        #pca = PCA(n_components=50).fit(X_DDTpd)
        #print(pca.explained_variance_ratio_)
        #X_DDTpd = pca.transform(X_DDTpd)
        #X_eval = pca.transform(X_eval)


        final_model = lgb.LGBMClassifier(**best_params_, random_state=self.args.seed)
        #cv_score_best = cross_val_score(final_model, X_DDTpd, self.Y_train_proba, cv=5, verbose=6)
        #print(cv_score_best.mean(), cv_score_best.std())




        final_model.fit(X_DDTpd, self.Y_train_proba)



        self.plot_feat_importance(final_model, features,
                                  self.path_save_model + "features_importances_LGBM_nbrefeat_"+str(len(features))+".png")
        y_pred = final_model.predict(X_eval)

        #print(self.nn_model_ref.outputs_pred_val[:,0].shape, y_pred.shape)
        #print(self.nn_model_ref.outputs_pred_val[:,0], y_pred)

        print()
        self.save_logs(self.path_save_model + "logs_lgbm_" + str(len(features)) + ".txt", y_pred, self.Y_eval_proba)
        print()

        same_output = self.nn_model_ref.outputs_pred_val[:,0] == y_pred

        #print(same_output)



        p2 = 100 * np.sum(same_output) / len(same_output)
        print()
        print()
        print("MATCHING PROPORTION: " + str(p2)+ " %")



        index_interext = np.logical_and(same_output, self.Y_eval_proba == y_pred)
        p22 = 100 * np.sum(index_interext) / len(index_interext)
        print("MATCHING PROPORTION AND EQUAL TO LABEL: " + str(p22)+ " %")
        print()
        cm = confusion_matrix(y_pred=y_pred, y_true=self.Y_eval_proba, normalize="true")
        res = np.array([accuracy_score(self.Y_eval_proba, y_pred), cm[0][0], cm[1][1], p2/100, p22/100])
        print()

        print("SUMMARY: [Acc, TPR, TNR, Matching, Matching +label] - ", res)
        print()
        np.save(self.path_save_model + "res_" + str(self.cpt) + ".npy",res )


        lgb.create_tree_digraph(final_model).save(directory=self.path_save_model, filename="tree_LGBM_nbrefeat_"+str(len(features))+".dot")
        os.system("dot -Tpng " + self.path_save_model + "tree_LGBM_nbrefeat_"+str(len(features))+".dot > " + self.path_save_model + "tree_LGBM_nbrefeat_"+str(len(features))+".png")
        del X_DDTpd
        self.importances = final_model.feature_importances_
        self.indices = np.argsort(self.importances)[::-1]
        with open(self.path_save_model + "features_impotances_order_nbrefeat_"+str(len(features))+".txt", "w") as file:
            file.write(str(np.array(features)[self.indices]) + str(self.importances[self.indices]))
            file.write("\n")
        if self.masks_infos_score is None:
            self.masks_infos_score = self.importances.copy()
            self.masks_infos_rank = np.array([np.where(self.indices==x)[0][0] for x in range(len(self.importances))])
        return final_model



    def classifier_lgbm(self):
        X_DDTpd = pd.DataFrame(data=self.X_train_proba, columns=self.table_of_truth.features_name)
        clf = self.classifier_lgbm_general(X_DDTpd, self.X_eval_proba, self.table_of_truth.features_name)
        return clf

    def classifier_lr(self):
        X_DDTpd = pd.DataFrame(data=self.X_train_proba, columns=self.table_of_truth.features_name)
        clf = SVC(kernel="linear", C=0.025,random_state=self.args.seed)
        clf.fit(X_DDTpd, self.Y_train_proba)
        y_pred = clf.predict(self.X_eval_proba)
        print(accuracy_score(self.Y_eval_proba, y_pred.round()))
        return clf






    def classifier_lgbm_retrict(self):
        indices = np.argsort(self.importances)[::-1][:self.args.keep_number_most_impactfull]
        X_DDTpd = pd.DataFrame(data=self.X_train_proba[:, indices],
                               columns=np.array(self.table_of_truth.features_name)[indices])
        clf = self.classifier_lgbm_general(X_DDTpd, self.X_eval_proba[:, indices], np.array(self.table_of_truth.features_name)[indices])


        with open(self.path_save_model + "masks_all_most_imp.txt", "w") as file:
            for i in range(len(self.args.inputs_type)):
                file.write(str(list(np.array(self.get_masks_gen.masks[i])[indices])))
                file.write("\n")

        return clf, indices


    def classifier_RF_general(self, X_DDTpd, X_eval, features):
        best_params_RF = {'n_estimators': 100,
                          'max_features': 'auto',
                          'max_depth': 100,
                          'min_samples_split': 5,
                          'min_samples_leaf': 2,
                          'bootstrap': True}
        final_model = RandomForestClassifier(**best_params_RF, random_state=self.args.seed)
        cv_score_best = cross_val_score(final_model, X_DDTpd, self.Y_train_proba, cv=5, verbose=6)
        print(cv_score_best.mean(), cv_score_best.std())
        final_model.fit(X_DDTpd, self.Y_train_proba)
        self.plot_feat_importance(final_model, features,
                                  self.path_save_model + "features_importances_RF_nbrefeat_"+str(len(features))+".png")
        y_pred = final_model.predict(X_eval)
        self.save_logs(self.path_save_model + "logs_RF_"+str(len(features))+".txt", y_pred, self.Y_eval_proba)

        export_graphviz(final_model.estimators_[5],
                        out_file=self.path_save_model + "tree_RF_nbrefeat_"+str(len(features))+".dot",
                        feature_names=features, class_names=["Random", "Speck"],
                        rounded=True, proportion=False, precision=2, filled=True)

        os.system("dot -Tpng " + self.path_save_model + "tree_RF_nbrefeat_"+str(len(features))+".dot > " + self.path_save_model + "tree_RF_nbrefeat_"+str(len(features))+".png")
        del X_DDTpd
        self.importances = final_model.feature_importances_
        indices = np.argsort(self.importances)[::-1]
        with open(self.path_save_model + "features_impotances_order_RF_nbrefeat_"+str(len(features))+".txt", "w") as file:
            file.write(str(np.array(features)[indices]) + str(self.importances[indices]))
            file.write("\n")
        return final_model


    def classifier_RF(self):
        X_DDTpd = pd.DataFrame(data=self.X_train_proba, columns=self.table_of_truth.features_name)
        clf = self.classifier_RF_general(X_DDTpd, self.X_eval_proba, self.table_of_truth.features_name)
        return clf


    def classifier_RF_retrict(self):
        indices = np.argsort(self.importances)[::-1][:self.args.keep_number_most_impactfull]
        X_DDTpd = pd.DataFrame(data=self.X_train_proba[:, indices],
                               columns=np.array(self.table_of_truth.features_name)[indices])
        clf = self.classifier_RF_general(X_DDTpd, self.X_eval_proba[:, indices], np.array(self.table_of_truth.features_name)[indices])
        return clf, indices




    def save_logs(self, path_save_model_txt, y_pred,Y_vf):
        with open(path_save_model_txt, 'w') as f:
            print("ACCURACY")
            f.write("ACCURACY")
            f.write("\n")
            print(accuracy_score(y_pred=y_pred, y_true=Y_vf))
            f.write(str(accuracy_score(y_pred=y_pred, y_true=Y_vf)))
            f.write("\n")
            print("Confusion matrix")
            f.write("Confusion matrix")
            f.write("\n")
            print(confusion_matrix(y_pred=y_pred, y_true=Y_vf))
            f.write(str(confusion_matrix(y_pred=y_pred, y_true=Y_vf)))
            f.write("\n")
            print(confusion_matrix(y_pred=y_pred, y_true=Y_vf, normalize="true"))
            f.write(str(confusion_matrix(y_pred=y_pred, y_true=Y_vf, normalize="true")))
            f.write("\n")
            print()
            print(metrics.classification_report(Y_vf, y_pred, target_names=["random", "speck"], digits=4))
            f.write(str(metrics.classification_report(Y_vf, y_pred, target_names=["random", "speck"], digits=4)))
            f.write("\n")
            print()
            print('Mean Absolute Error:', metrics.mean_absolute_error(Y_vf, y_pred))
            print('Mean Squared Error:', metrics.mean_squared_error(Y_vf, y_pred))
            print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_vf, y_pred)))
            f.write('Mean Absolute Error: '+ str(metrics.mean_absolute_error(Y_vf, y_pred)))
            f.write("\n")
            f.write('Mean Squared Error: '+ str(metrics.mean_squared_error(Y_vf, y_pred)))
            f.write("\n")
            f.write('Root Mean Squared Error: '+ str(np.sqrt(metrics.mean_squared_error(Y_vf, y_pred))))
            f.close()

    def plot_feat_importance(self, final_model, feature_name, name):
        importances = final_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        colnames = feature_name
        fig, ax = plt.subplots(1, 1, figsize=(75, 75))
        ax.set_title("Feature importances")
        ax.barh(range(len(colnames)), importances[indices[::-1]],
                color="r", align="center")
        ax.set_yticks(range(len(colnames)))
        ax.set_yticklabels(np.array(colnames)[indices][::-1])
        plt.savefig(name)



def evaluate_all(all_clfs, generator_data, nn_model_ref, table_of_truth, qm, path_save_model):
    generator_data.create_data_bin_val()
    generator_data.create_data_g_val(table_of_truth)
    nn_model_ref.X_train_nn_binaire = generator_data.X_bin_train
    nn_model_ref.X_val_nn_binaire = generator_data.X_bin_val
    nn_model_ref.Y_train_nn_binaire = generator_data.Y_create_proba_train
    nn_model_ref.Y_val_nn_binaire = generator_data.Y_create_proba_val
    nn_model_ref.eval_all(["val"])



    X_eval_proba = generator_data.X_proba_val
    Y_eval_proba = generator_data.Y_create_proba_val

    columns_1 = list(all_clfs.clf_final.keys())
    columns_2 = [x + " prediction proba" for x in columns_1]
    columns_3 = [x + " prediction boolean" for x in columns_1]
    columns_4 = ["label"]
    columns_5 = ["NN_ref prediction proba", "NN_ref prediction boolean"]
    columns = columns_2 + columns_3 + columns_4 + columns_5
    index = [x for x in range(X_eval_proba.shape[0])]
    results_all = pd.DataFrame(index=index, columns=columns)
    results_all["label"] = Y_eval_proba

    key = "NN_ref"
    results_all[key + " prediction proba"] = nn_model_ref.outputs_proba_val
    results_all[key + " prediction boolean"] = nn_model_ref.outputs_pred_val

    del generator_data, table_of_truth, qm

    if "NN_ref_retrain" in columns_1:
        X_eval_proba_feat = nn_model_ref.all_intermediaire_val
        clf = all_clfs.clf_final["NN_ref_retrain"]
        predictions = clf.predict(X_eval_proba_feat)
        predictions_acc = predictions > 0.5
        predictions_acc = predictions_acc.astype(int)
        print("ACCURACY MODEL " + str("NN_ref_retrain") + " : ",
              accuracy_score(y_pred=predictions_acc, y_true=Y_eval_proba))
        key = "NN_ref_retrain"
        results_all[key + " prediction proba"] = predictions
        results_all[key + " prediction boolean"] = predictions_acc

    del nn_model_ref

    for key in columns_1:
        clf = all_clfs.clf_final[key]
        if "NN" == key:
            met2_predictions = clf.predict(X_eval_proba)
            met2_predictions_acc = met2_predictions > 0.5
            met2_predictions_acc = met2_predictions_acc.astype(int)
            print("ACCURACY "+str(key)+" OUR : ", accuracy_score(y_pred=met2_predictions_acc, y_true=Y_eval_proba))
            results_all[key + " prediction proba"] = met2_predictions
            results_all[key + " prediction boolean"] = (met2_predictions_acc )
        if "LGBM" == key:
            met2_predictions = clf.predict(X_eval_proba)
            met2_predictions = np.expand_dims(met2_predictions, axis=1)

            met2_predictions_p = clf.predict_proba(X_eval_proba)
            

            met2_predictions_acc = met2_predictions > 0.5
            met2_predictions_acc = met2_predictions_acc.astype(int)
            print("ACCURACY "+str(key)+" OUR : ", accuracy_score(y_pred=met2_predictions_acc, y_true=Y_eval_proba))
            results_all[key + " prediction proba"] =  1- met2_predictions_p
            results_all[key + " prediction boolean"] = (met2_predictions_acc )
        if "RF" == key:
            met2_predictions = clf.predict(X_eval_proba)
            met2_predictions = np.expand_dims(met2_predictions, axis=1)
            met2_predictions_p = clf.predict_proba(X_eval_proba)
            
            met2_predictions_acc = met2_predictions > 0.5
            met2_predictions_acc = met2_predictions_acc.astype(int)
            print("ACCURACY "+str(key)+" OUR : ", accuracy_score(y_pred=met2_predictions_acc, y_true=Y_eval_proba))
            results_all[key + " prediction proba"] = met2_predictions_p
            results_all[key + " prediction boolean"] = (met2_predictions_acc  )
        if "LGBM_restricted" == key:
            clf2, indices = clf[0], clf[1]
            met2_predictions = clf2.predict(X_eval_proba[:, indices])
            met2_predictions = np.expand_dims(met2_predictions, axis=1)

            met2_predictions_p = clf2.predict_proba(X_eval_proba[:, indices])
            

            met2_predictions_acc = met2_predictions > 0.5
            met2_predictions_acc = met2_predictions_acc.astype(int)
            print("ACCURACY "+str(key)+" OUR : ", accuracy_score(y_pred=met2_predictions_acc, y_true=Y_eval_proba))
            results_all[key + " prediction proba"] = met2_predictions_p
            results_all[key + " prediction boolean"] = (met2_predictions_acc )
        if "RF_restricted" == key:
            clf2, indices = clf[0], clf[1]
            met2_predictions = clf2.predict(X_eval_proba[:, indices])
            met2_predictions = np.expand_dims(met2_predictions, axis=1)

            met2_predictions_p = clf2.predict_proba(X_eval_proba[:, indices])
            

            met2_predictions_acc = met2_predictions > 0.5
            met2_predictions_acc = met2_predictions_acc.astype(int)
            print("ACCURACY "+str(key)+" OUR : ", accuracy_score(y_pred=met2_predictions_acc, y_true=Y_eval_proba))
            results_all[key + " prediction proba"] = met2_predictions_p
            results_all[key + " prediction boolean"] = (met2_predictions_acc )
    results_all.to_csv(path_save_model + "results_finaux.csv", index=False)