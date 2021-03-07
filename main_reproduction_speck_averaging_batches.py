import os

for seed in range(8, 11):
    for r in [5, 6, 7]:
        for N in [1, 5, 10, 50, 100]:
            if r==5:
                load_nn_path = "./models/speck_5round_ND_model_inputv2.pth"
            elif r==6:
                load_nn_path = "./models/speck_6round_ND_model_inputv2.pth"
            nbre_sample_eval = int(10**6/N)
            if r==5 and N>11:
                print("pass")
            elif r == 6 and N > 51:
                print("pass")
            else:
                print("Start config: ", seed, r, N, nbre_sample_eval)
                print("Seed: ", seed)
                print("Round: ", r)
                print("Number of batch: ", N)
                print("Number of samples: ", nbre_sample_eval)
                if r==7:
                    os.system("python3 eval_Nbatches.py --nbre_sample_eval " + str(nbre_sample_eval) + " --Nbatch "
                              + str(N) + " --nombre_round_eval " + str(r) + " --seed " + str(seed))
                else:
                    os.system("python3 eval_Nbatches.py --nbre_sample_eval " + str(nbre_sample_eval) + " --Nbatch "
                              + str(N) + " --nombre_round_eval " + str(r) + " --seed " + str(seed)
                              + " --load_nn_path " + str(load_nn_path) + " --type_model " + "baseline_real"
                              + " --load_special Yes")