from src.classifiers.nn_classifier_keras import train_speck_distinguisher


class All_classifier:


    def __init__(self, args, generator_data, nn_model_ref, path_save_model, get_masks_gen):
        self.args = args
        self.generator_data = generator_data
        self.nn_model_ref = nn_model_ref
        self.get_masks_gen = get_masks_gen
        self.path_save_model = path_save_model
        self.X_train_proba = generator_data.X_proba_train
        self.Y_train_proba = generator_data.Y_create_proba_train
        self.X_eval_proba = generator_data.X_proba_val
        self.Y_eval_proba = generator_data.Y_create_proba_val
        if args.retrain_nn_ref:
            self.retrain_nn_ref(nn_model_ref)



    def retrain_nn_ref(self):
        self.nn_model_ref.epochs = self.args.num_epch_2
        self.nn_model_ref.batch_size_2 = self.args.batch_size_2
        self.nn_model_ref.net.freeze()
        X_train_proba_feat, X_eval_proba_feat = self.nn_model_ref.all_intermediaire, self.nn_model_ref.all_intermediaire_val
        Y_train_proba = self.generator_data.Y_create_proba_train
        Y_eval_proba = self.generator_data.Y_create_proba_val
        print("START RETRAIN LINEAR NN GOHR ")
        print()
        net_retrain, h = train_speck_distinguisher(self.args, X_train_proba_feat.shape[1], X_train_proba_feat,
                                                   Y_train_proba, X_eval_proba_feat, Y_eval_proba,
                                                   bs=self.args.batch_size_2,
                                                   epoch=self.args.num_epch_2, name_ici="retrain_nn_gohr",
                                                   wdir=self.path_save_model)

    def train_classifier_nn(self):
        net_our, h = train_speck_distinguisher(self.args, len(self.get_masks_gen.masks[0]), self.X_train_proba,
                                            self.Y_train_proba, self.X_eval_proba, self.Y_eval_proba, bs=self.args.batch_size_our,
                                            epoch=self.args.num_epch_our, name_ici="our_model",
                                            wdir=self.path_save_model)

    def train_classifier_LGBM(self):
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

        X_DDTpd = pd.DataFrame(data=self.X_train_proba, columns=self.get_masks_gen.fe)
        final_model = lgb.LGBMClassifier(**best_params_, random_state=args.seed)
        final_model.fit(X_DDTpd, Y_tf)