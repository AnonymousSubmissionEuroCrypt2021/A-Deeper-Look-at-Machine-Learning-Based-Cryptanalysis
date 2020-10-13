import random
import torch
from torch.utils.tensorboard import SummaryWriter
import json
import datetime
import numpy as np
from pathlib import Path

from src.data_cipher.cipher_simeck import Cipher_Simeck
from src.data_cipher.cipher_simon import Cipher_Simon
from src.data_cipher.cipher_speck import Cipher_Speck


def init_all_for_run(args):
    date = str(datetime.datetime.now()).replace(" ", "_").replace("-", "_").replace(":", "_").replace(".", "_")
    name_input = ""
    for x in args.inputs_type:
        name_input+=str(x)
        name_input+="_"
    name_input = name_input[:-1]
    slsh = "/"
    path_save_model = args.logs_tensorboard +  args.cipher + slsh +str(args.nombre_round_eval) +slsh + name_input + slsh + date + slsh
    path_save_model_train = args.models_path +  args.cipher + slsh +str(args.nombre_round_eval) +slsh + name_input + slsh
    print()
    print("Folder save: ", path_save_model)
    writer = SummaryWriter(path_save_model)
    with open(path_save_model+'commandline_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    Path(path_save_model_train).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:"+str(args.device) if torch.cuda.is_available() else "cpu")
    print()
    print("Use Hardware : ", device)
    print()

    # Reproductibilites
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    rng = random.Random(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return writer, device, rng, path_save_model, path_save_model_train, name_input

def init_cipher(args):
    if args.cipher == "speck":
        cipher = Cipher_Speck(args)
    if args.cipher == "simon":
        cipher = Cipher_Simon(args)
    if args.cipher == "simeck":
        cipher = Cipher_Simeck(args)
    if args.cipher == "aes228":
        cipher = Cipher_aes_228(args)
    if args.cipher == "aes224":
        cipher = Cipher_aes_224(args)
    return cipher