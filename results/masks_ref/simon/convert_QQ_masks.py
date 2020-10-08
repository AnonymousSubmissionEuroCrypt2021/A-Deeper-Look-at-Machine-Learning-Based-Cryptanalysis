import numpy as np
liste_all_mask = []
masksc0l = []
masksc0r = []
masksc1l = []
masksc1r = []
infile = open(
    "XOR_correlation_7_gooddiff.txt",
            'r')
for line in infile:
    masks = []
    print(line)
    line2 = line.split(")")[0].split("(")[-1].split(", ")
    print(line2)
    for one in range(64):
        if str(one) in line2:
            masks.append(1)
        else:
            masks.append(0)

    c0l = int("".join(str(i) for i in masks[:16]),2)
    c0r = int("".join(str(i) for i in masks[16:32]),2)
    c1l = int("".join(str(i) for i in masks[32:48]),2)
    c1r = int("".join(str(i) for i in masks[48:]),2)
    print(masks, masks[48:])
    nc0 = np.sum(np.array([int(x) for x in list('{0:0b}'.format(c0l))]))
    nc1 = np.sum(np.array([int(x) for x in list('{0:0b}'.format(c0r))]))
    nc2 = np.sum(np.array([int(x) for x in list('{0:0b}'.format(c1l))]))
    nc3 = np.sum(np.array([int(x) for x in list('{0:0b}'.format(c1r))]))


    if (c0l, c0r, c1l, c1r) not in liste_all_mask:
        liste_all_mask.append((c0l, c0r, c1l, c1r))
        masksc0l.append(c0l)
        masksc0r.append(c0r)
        masksc1l.append(c1l)
        masksc1r.append(c1r)

infile.close()


with open("masks_"+str(len(masksc0l))+".txt","w") as file:
    file.write(str(masksc0l))
    file.write("\n")
    file.write(str(masksc0r))
    file.write("\n")
    file.write(str(masksc1l))
    file.write("\n")
    file.write(str(masksc1r))
    file.write("\n")