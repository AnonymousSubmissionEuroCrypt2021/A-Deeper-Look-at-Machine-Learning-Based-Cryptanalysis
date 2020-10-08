import sys
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
from collections import Counter
from src.get_masks.evaluate_quality_masks import Quality_masks
from src.classifiers.classifier_all import All_classifier, evaluate_all
from src.ToT.table_of_truth import ToT
from src.data_classifier.Generator_proba_classifier import Genrator_data_prob_classifier
from src.get_masks.get_masks import Get_masks
from src.nn.nn_model_ref import NN_Model_Ref
from src.data_cipher.create_data import Create_data_binary
from src.utils.initialisation_run import init_all_for_run, init_cipher
from src.utils.config import Config
import argparse
from src.utils.utils import str2bool, two_args_str_int, two_args_str_float, str2list, transform_input_type, str2hexa
import numpy as np
import pandas as pd
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# initiate the parser

print("TODO: MULTITHREADING + ASSERT PATH EXIST DEPEND ON CONDITION + save DDT + deleate some dataset")

config = Config(path="config_reproduction_3/")
parser = argparse.ArgumentParser()

parser.add_argument("--seed", default=config.general.seed, type=two_args_str_int, choices=[i for i in range(100)])
parser.add_argument("--device", default=config.general.device, type=two_args_str_int, choices=[i for i in range(100)])
parser.add_argument("--logs_tensorboard", default=config.general.logs_tensorboard)
parser.add_argument("--models_path", default=config.general.models_path)
parser.add_argument("--models_path_load", default=config.general.models_path_load)
parser.add_argument("--cipher", default=config.general.cipher, choices=["speck", "simon", "aes228", "aes224", "simeck", "gimli"])
parser.add_argument("--nombre_round_eval", default=config.general.nombre_round_eval, type=two_args_str_int)
parser.add_argument("--inputs_type", default=config.general.inputs_type, type=transform_input_type)
parser.add_argument("--word_size", default=config.general.word_size, type=two_args_str_int)
parser.add_argument("--alpha", default=config.general.alpha, type=two_args_str_int)
parser.add_argument("--beta", default=config.general.beta, type=two_args_str_int)
parser.add_argument("--type_create_data", default=config.general.type_create_data, choices=["normal", "real_difference"])

parser.add_argument("--diff", default=config.train_nn.diff, type=str2hexa)
parser.add_argument("--retain_model_gohr_ref", default=config.train_nn.retain_model_gohr_ref, type=str2bool)
parser.add_argument("--load_special", default=config.train_nn.load_special, type=str2bool)
parser.add_argument("--finetunning", default=config.train_nn.finetunning, type=str2bool)
parser.add_argument("--model_finetunne", default=config.train_nn.model_finetunne, choices=["baseline", "cnn_attention", "multihead", "deepset", "baseline_bin"])
parser.add_argument("--load_nn_path", default=config.train_nn.load_nn_path)
parser.add_argument("--countinuous_learning", default=config.train_nn.countinuous_learning, type=str2bool)
parser.add_argument("--curriculum_learning", default=config.train_nn.curriculum_learning, type=str2bool)
parser.add_argument("--nbre_epoch_per_stage", default=config.train_nn.nbre_epoch_per_stage, type=two_args_str_int)
parser.add_argument("--a_bit", default=config.train_nn.a_bit, type=two_args_str_int)
parser.add_argument("--type_model", default=config.train_nn.type_model, choices=["baseline", "cnn_attention", "multihead", "deepset", "baseline_bin", "baseline_bin_v2"])
parser.add_argument("--nbre_sample_train", default=config.train_nn.nbre_sample_train, type=two_args_str_int)
parser.add_argument("--nbre_sample_eval", default=config.train_nn.nbre_sample_eval, type=two_args_str_int)
parser.add_argument("--num_epochs", default=config.train_nn.num_epochs, type=two_args_str_int)
parser.add_argument("--batch_size", default=config.train_nn.batch_size, type=two_args_str_int)
parser.add_argument("--loss_type", default=config.train_nn.loss_type, choices=["BCE", "MSE", "SmoothL1Loss", "CrossEntropyLoss", "F1"])
parser.add_argument("--lr_nn", default=config.train_nn.lr_nn, type=two_args_str_float)
parser.add_argument("--weight_decay_nn", default=config.train_nn.weight_decay_nn, type=two_args_str_float)
parser.add_argument("--momentum_nn", default=config.train_nn.momentum_nn, type=two_args_str_float)
parser.add_argument("--base_lr", default=config.train_nn.base_lr, type=two_args_str_float)
parser.add_argument("--max_lr", default=config.train_nn.max_lr, type=two_args_str_float)
parser.add_argument("--demicycle_1", default=config.train_nn.demicycle_1, type=two_args_str_int)
parser.add_argument("--optimizer_type", default=config.train_nn.optimizer_type, choices=["Adam", "AdamW", "SGD"])
parser.add_argument("--scheduler_type", default=config.train_nn.scheduler_type, choices=["CyclicLR", "None"])
parser.add_argument("--numLayers", default=config.train_nn.numLayers, type=two_args_str_int)
parser.add_argument("--limit", default=config.train_nn.limit, type=two_args_str_int)
parser.add_argument("--kstime", default=config.train_nn.kstime, type=two_args_str_int)
parser.add_argument("--out_channel0", default=config.train_nn.out_channel0, type=two_args_str_int)
parser.add_argument("--out_channel1", default=config.train_nn.out_channel1, type=two_args_str_int)
parser.add_argument("--hidden1", default=config.train_nn.hidden1, type=two_args_str_int)
parser.add_argument("--hidden2", default=config.train_nn.hidden2, type=two_args_str_int)
parser.add_argument("--kernel_size0", default=config.train_nn.kernel_size0, type=two_args_str_int)
parser.add_argument("--kernel_size1", default=config.train_nn.kernel_size1, type=two_args_str_int)
parser.add_argument("--num_workers", default=config.train_nn.num_workers, type=two_args_str_int)
parser.add_argument("--clip_grad_norm", default=config.train_nn.clip_grad_norm, type=two_args_str_float)
parser.add_argument("--end_after_training", default=config.train_nn.end_after_training, type=str2bool)



parser.add_argument("--load_masks", default=config.getting_masks.load_masks, type=str2bool)
parser.add_argument("--file_mask", default=config.getting_masks.file_mask)
parser.add_argument("--nbre_max_masks_load", default=config.getting_masks.nbre_max_masks_load, type=two_args_str_int)
parser.add_argument("--nbre_generate_data_train_val", default=config.getting_masks.nbre_generate_data_train_val, type=two_args_str_int)
parser.add_argument("--nbre_necessaire_val_SV", default=config.getting_masks.nbre_necessaire_val_SV, type=two_args_str_int)
parser.add_argument("--nbre_max_batch", default=config.getting_masks.nbre_max_batch, type=two_args_str_int)
parser.add_argument("--liste_segmentation_prediction", default=config.getting_masks.liste_segmentation_prediction)
parser.add_argument("--liste_methode_extraction", default=config.getting_masks.liste_methode_extraction, type=transform_input_type)
parser.add_argument("--liste_methode_selection", default=config.getting_masks.liste_methode_selection, type=transform_input_type)
parser.add_argument("--hamming_weigth", default=config.getting_masks.hamming_weigth, type=str2list)
parser.add_argument("--thr_value", default=config.getting_masks.thr_value, type=str2list)
parser.add_argument("--research_new_masks", default=config.getting_masks.research_new_masks, type=str2bool)
parser.add_argument("--save_fig_plot_feature_before_mask", default=config.getting_masks.save_fig_plot_feature_before_mask, type=str2bool)
parser.add_argument("--end_after_step2", default=config.getting_masks.end_after_step2, type=str2bool)


parser.add_argument("--create_new_data_for_ToT", default=config.make_ToT.create_new_data_for_ToT, type=str2bool)
parser.add_argument("--create_ToT_with_only_sample_from_cipher", default=config.make_ToT.create_ToT_with_only_sample_from_cipher, type=str2bool)
parser.add_argument("--nbre_sample_create_ToT", default=config.make_ToT.nbre_sample_create_ToT, type=two_args_str_int)

parser.add_argument("--create_new_data_for_classifier", default=config.make_data_classifier.create_new_data_for_classifier, type=str2bool)
parser.add_argument("--nbre_sample_train_classifier", default=config.make_data_classifier.nbre_sample_train_classifier, type=two_args_str_int)
parser.add_argument("--nbre_sample_val_classifier", default=config.make_data_classifier.nbre_sample_val_classifier, type=two_args_str_int)

parser.add_argument("--retrain_nn_ref", default=config.compare_classifer.retrain_nn_ref, type=str2bool)
parser.add_argument("--num_epch_2", default=config.compare_classifer.num_epch_2, type=two_args_str_int)
parser.add_argument("--batch_size_2", default=config.compare_classifer.batch_size_2, type=two_args_str_int)
parser.add_argument("--classifiers_ours", default=config.compare_classifer.classifiers_ours, type=str2list)
parser.add_argument("--retrain_with_import_features", default=config.compare_classifer.retrain_with_import_features, type=str2bool)
parser.add_argument("--keep_number_most_impactfull", default=config.compare_classifer.keep_number_most_impactfull, type=two_args_str_int)
parser.add_argument("--num_epch_our", default=config.compare_classifer.num_epch_our, type=two_args_str_int)
parser.add_argument("--batch_size_our", default=config.compare_classifer.batch_size_our, type=two_args_str_int)
parser.add_argument("--alpha_test", default=config.compare_classifer.alpha_test, type=two_args_str_float)
parser.add_argument("--quality_of_masks", default=config.compare_classifer.quality_of_masks, type=str2bool)
parser.add_argument("--end_after_step4", default=config.compare_classifer.end_after_step4, type=str2bool)
parser.add_argument("--eval_nn_ref", default=config.compare_classifer.eval_nn_ref, type=str2bool)
parser.add_argument("--compute_independance_feature", default=config.compare_classifer.compute_independance_feature, type=str2bool)
parser.add_argument("--save_data_proba", default=config.compare_classifer.save_data_proba, type=str2bool)



args = parser.parse_args()


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#
print("---" * 100)
writer, device, rng, path_save_model, path_save_model_train, name_input = init_all_for_run(args)


print("LOAD CIPHER")
print()
cipher = init_cipher(args)

cpt = 0


for seed in range(3):
    args.seed = seed
    creator_data_binary = Create_data_binary(args, cipher, rng)
    for repeat in range(5):
        cpt+=1
        creator_data_binary = Create_data_binary(args, cipher, rng)

        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        print("---" * 100)
        print("STEP 1 : LOAD/ TRAIN NN REF")
        print()
        print("COUNTINUOUS LEARNING: "+ str(args.countinuous_learning) +  " | CURRICULUM LEARNING: " +  str(args.curriculum_learning) + " | MODEL: " + str(args.type_model))
        print()



        nn_model_ref = NN_Model_Ref(args, writer, device, rng, path_save_model, cipher, creator_data_binary, path_save_model_train)


        if args.retain_model_gohr_ref:
            nn_model_ref.train_general(name_input)
        else:
            #nn_model_ref.load_nn()
            try:
                if args.finetunning:
                    nn_model_ref.load_nn()
                    nn_model_ref.train_from_scractch(name_input + "fine-tune")
                #nn_model_ref.eval(["val"])
                else:
                    nn_model_ref.load_nn()
            except:
                print("ERROR")
                print("NO MODEL AVALAIBLE FOR THIS CONFIG")
                print("CHANGE ARGUMENT retain_model_gohr_ref")
                print()
                sys.exit(1)

        if args.create_new_data_for_ToT and args.create_new_data_for_classifier:
            del nn_model_ref.X_train_nn_binaire, nn_model_ref.X_val_nn_binaire, nn_model_ref.Y_train_nn_binaire, nn_model_ref.Y_val_nn_binaire
            del nn_model_ref.c0l_train_nn, nn_model_ref.c0l_val_nn, nn_model_ref.c0r_train_nn, nn_model_ref.c0r_val_nn
            del nn_model_ref.c1l_train_nn, nn_model_ref.c1l_val_nn, nn_model_ref.c1r_train_nn, nn_model_ref.c1r_val_nn


        print("STEP 1 : DONE")
        print("---" * 100)
        if args.end_after_training:
            sys.exit(1)

        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        print("STEP 2 : GET MASKS")
        print()
        print("LOAD MASKS: "+ str(args.load_masks) +  " | RESEARCH NEW: " +  str(args.research_new_masks) + " | MODEL: " + str(args.liste_segmentation_prediction) +" " + str(args.liste_methode_extraction)+" " + str(args.liste_methode_selection)+" " + str(args.hamming_weigth)+" " + str(args.thr_value))
        print()
        get_masks_gen = Get_masks(args, nn_model_ref.net, path_save_model, rng, creator_data_binary, device)
        if args.research_new_masks:
            get_masks_gen.start_step()
            del get_masks_gen.X_deltaout_train, get_masks_gen.X_eval, get_masks_gen.Y_tf, get_masks_gen.Y_eval



        print("STEP 2 : DONE")
        print("---" * 100)
        if args.end_after_step2:
            sys.exit(1)

        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        print("STEP 3 : MAKE TABLE OF TRUTH")
        print()
        print("NEW DATA: "+ str(args.create_new_data_for_ToT) +  " | PURE ToT: " +  str(args.create_ToT_with_only_sample_from_cipher) )
        print()

        table_of_truth = ToT(args, nn_model_ref.net, path_save_model, rng, creator_data_binary, device, get_masks_gen.masks, nn_model_ref)
        table_of_truth.create_DDT()

        del table_of_truth.c0l_create_ToT, table_of_truth.c0r_create_ToT
        del table_of_truth.c1l_create_ToT, table_of_truth.c1r_create_ToT

        print("STEP 3 : DONE")
        print("---" * 100)
        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        print("STEP 4 : CREATE DATA PROBA AND CLASSIFY")
        print()
        print("NEW DATA: "+ str(args.create_new_data_for_classifier))
        print()

        generator_data = Genrator_data_prob_classifier(args, nn_model_ref.net, path_save_model, rng, creator_data_binary, device, get_masks_gen.masks, nn_model_ref)
        generator_data.create_data_g(table_of_truth)
        #EVALUATE GOHR NN ON NEW DATASET
        nn_model_ref.X_train_nn_binaire = generator_data.X_bin_train
        nn_model_ref.X_val_nn_binaire = generator_data.X_bin_val
        nn_model_ref.Y_train_nn_binaire = generator_data.Y_create_proba_train
        nn_model_ref.Y_val_nn_binaire = generator_data.Y_create_proba_val

        if args.eval_nn_ref:
            nn_model_ref.eval_all(["train", "val"])
        all_clfs = All_classifier(args, path_save_model, generator_data, get_masks_gen, nn_model_ref, table_of_truth, cpt)
        #all_clfs.X_train_proba = np.concatenate((all_clfs.X_train_proba, X_feat_temp), axis = 1)
        #all_clfs.X_eval_proba =  np.concatenate((all_clfs.X_eval_proba, X_feat_temp_val), axis = 1)
        all_clfs.classify_all()

        if args.quality_of_masks:
            qm = Quality_masks(args, path_save_model, generator_data, get_masks_gen, nn_model_ref, table_of_truth, all_clfs)
            qm.start_all()
        else:
            qm = None

        del nn_model_ref, all_clfs, generator_data, table_of_truth, get_masks_gen



