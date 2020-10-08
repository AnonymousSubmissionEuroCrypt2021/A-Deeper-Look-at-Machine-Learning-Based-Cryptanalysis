import os

#'[ctdata0l, ctdata0r, ctdata1l, ctdata1r]', '[ctdata0l^ctdata1l, ctdata0r^ctdata1r^ctdata0l^ctdata1l, ctdata0l^ctdata0r]', '[ctdata0l^ctdata1l, ctdata0r^ctdata1r^ctdata0l^ctdata1l, ctdata0l^ctdata0r, ctdata1l^ctdata1r]', '[ctdata0l^ctdata1l, ctdata0r^ctdata1r^ctdata0l^ctdata1l]'
#["baseline", "deepset", "cnn_attention", "multihead"]

"""
command = "python3 main_3class.py --nombre_round_eval 5 --type_model baseline_3class --make_data_equilibre_3class No"
os.system(command)

command = "python3 main_3class.py --nombre_round_eval 6 --type_model baseline_3class --make_data_equilibre_3class No"
os.system(command)

command = "python3 main_3class.py --nombre_round_eval 5 --type_model baseline_3class"
os.system(command)

command = "python3 main_3class.py --nombre_round_eval 6 --type_model baseline_3class"
os.system(command)

command = "python3 main_3class.py --nombre_round_eval 5 --inputs_type '[ctdata0l, ctdata0r, ctdata1l, ctdata1r]' --type_model baseline_3class --make_data_equilibre_3class No"
os.system(command)

command = "python3 main_3class.py --nombre_round_eval 6 --inputs_type '[ctdata0l, ctdata0r, ctdata1l, ctdata1r]' --type_model baseline_3class --make_data_equilibre_3class No"
os.system(command)

command = "python3 main_3class.py --nombre_round_eval 5 --inputs_type '[ctdata0l, ctdata0r, ctdata1l, ctdata1r]' --type_model baseline_3class"
os.system(command)

command = "python3 main_3class.py --nombre_round_eval 6 --inputs_type '[ctdata0l, ctdata0r, ctdata1l, ctdata1r]' --type_model baseline_3class"
os.system(command)




command = "python3 main.py --inputs_type '[ctdata0l, ctdata0r, ctdata1l, ctdata1r]' --nombre_round_eval 5"
os.system(command)
command = "python3 main.py --inputs_type '[ctdata0l, ctdata0r, ctdata1l, ctdata1r]' --nombre_round_eval 6"
os.system(command)
#command = "python3 main.py --inputs_type '[ctdata0l, ctdata0r, ctdata1l, ctdata1r]' --nombre_round_eval 7"
#os.system(command)

command = "python3 main.py --nombre_round_eval 5"
os.system(command)
command = "python3 main.py --nombre_round_eval 6"
os.system(command)
#command = "python3 main.py --nombre_round_eval 7"
#os.system(command)

command = "python3 main.py --inputs_type '[ctdata0l, ctdata0r, ctdata1l, ctdata1r]' --type_model baseline_bin_v2 --nombre_round_eval 5"
os.system(command)
command = "python3 main.py --inputs_type '[ctdata0l, ctdata0r, ctdata1l, ctdata1r]' --type_model baseline_bin_v2 --nombre_round_eval 6"
os.system(command)
#command = "python3 main.py --inputs_type '[ctdata0l, ctdata0r, ctdata1l, ctdata1r]' --type_model baseline_bin_v2 --nombre_round_eval 7"
#os.system(command)

command = "python3 main.py --type_model baseline_bin_v2 --nombre_round_eval 5"
os.system(command)
command = "python3 main.py --type_model baseline_bin_v2 --nombre_round_eval 6"
os.system(command)
#command = "python3 main.py --type_model baseline_bin_v2 --nombre_round_eval 7"
#os.system(command)

"""

"""for r in [5, 6, 7]:
    #command = "python3 main.py --nombre_round_eval "+str(r)
    #os.system(command)
    for N in [50, 100]:
        Ntrain = 10**7
        Ntrain2 = int(Ntrain/N)
        Nval2 = int(Ntrain/(10*N))
        command = "python3 main_N_batch.py --Nbatch "+str(N) + " --nombre_round_eval "+str(r+1) + " --nbre_sample_train " + str(Ntrain2)+ " --nbre_sample_eval " + str(Nval2)
        os.system(command)"""


"""for r in [5, 4, 3, 2]:
    for lks in [[1,9], [2,12], [3,15]]:
        command = "python3 main.py --numLayers "+str(r) + " --limit "+str(lks[0]) + " --kstime "+str(lks[1])
        os.system(command)"""


#self.diff = (0x8100, 0x8102) #98.4 sur 3 round
#self.diff = (0x8300, 0x8302) #XXX sur 3 round
#self.diff = (0x8700, 0x8702) #91.7 sur 3 round
#self.diff = (0x8f00, 0x8f02) #XXX sur 3 round
#self.diff = (0x9f00, 0x9f02) #XXX sur 3 round
#self.diff = (0xbf00, 0xbf02) #XXX sur 3 round
#self.diff = (0xff00, 0xff02) #XXX sur 3 round
#self.diff = (0x7f00, 0x7f02) #XXX sur 3 round

for seed in range(8, 11):
    for r in [5, 6, 7]:
        for N in [1, 5, 10, 50, 100]:
            nbre_sample_eval = int(10**6/N)
            if r==5 and N>11:
                print("pass")
            elif r == 6 and N > 51:
                print("pass")
            else:
                print(r, N)
                os.system("python3 eval_Nbatches.py --nbre_sample_eval "+str(nbre_sample_eval)+" --Nbatch " + str(N) + " --nombre_round_eval " + str(r)+ " --seed " + str(seed))
