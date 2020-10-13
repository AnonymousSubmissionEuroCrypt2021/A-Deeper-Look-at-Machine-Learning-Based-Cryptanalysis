import numpy as np
import sys

class Create_data_binary:


    def __init__(self, args, cipher, rng):
        self.args = args
        self.cipher = cipher
        self.rng = rng
        self.WORD_SIZE = self.args.word_size

        self.diffs = [(0x8100, 0x8102), (0x8300, 0x8302), (0x8700, 0x8702), (0x8f00, 0x8f02), (0x9f00, 0x9f02),
                      (0xbf00, 0xbf02),(0xff00, 0xff02), (0x7f00, 0x7f02)]

        self.ps = [1/2, (1/2)**2, (1/2)**3, (1/2)**4, (1/2)**5, (1/2)**6, (1/2)**7,(1/2)**7]

        if args.cipher == "speck":
            #round0
            self.diff = args.diff #(0x8100, 0x8102)

            #self.diff = (0x0040, 0x0000)
            # roun1
            #self.diff = (0x8000, 0x8000)
            # round2
            #self.diff = (0x8100, 0x8102) #98.4 sur 3 round
            #self.diff = (0x8300, 0x8302) #XXX sur 3 round
            #self.diff = (0x8700, 0x8702) #91.7 sur 3 round
            #self.diff = (0x8f00, 0x8f02) #XXX sur 3 round
            #self.diff = (0x9f00, 0x9f02) #XXX sur 3 round
            #self.diff = (0xbf00, 0xbf02) #XXX sur 3 round
            #self.diff = (0xff00, 0xff02) #XXX sur 3 round
            #self.diff = (0x7f00, 0x7f02) #XXX sur 3 round
        if args.cipher == "simon":
            self.diff = (0, 0x0040)





        if args.cipher == "simeck":
            self.diff = (0x0, 0x2)
        if args.cipher == "aes228":
            self.diff = (1, 0, 0, 1)
        if args.cipher == "aes224":
            self.diff = (1, 0, 0, 1)


    def urandom_from_random(self, length):
        if length == 0:
            return b''
        chunk_size = 65535
        chunks = []
        while length >= chunk_size:
            chunks.append(self.rng.getrandbits(
                    chunk_size * 8).to_bytes(chunk_size, sys.byteorder))
            length -= chunk_size
        if length:
            chunks.append(self.rng.getrandbits(
                    length * 8).to_bytes(length, sys.byteorder))
        result = b''.join(chunks)
        return result


    def convert_to_binary(self, arr):
        X = np.zeros((len(arr) * self.WORD_SIZE, len(arr[0])), dtype=np.uint8);
        for i in range(len(arr) * self.WORD_SIZE):
            index = i // self.WORD_SIZE;
            offset = self.WORD_SIZE - (i % self.WORD_SIZE) - 1;
            X[i] = (arr[index] >> offset) & 1;
        X = X.transpose();
        return (X);


    def make_data(self, n):
        if self.args.cipher != "aes224" and self.args.cipher != "aes228" :
            X, Y, ctdata0l, ctdata0r, ctdata1l, ctdata1r = self.make_train_data_general(n)
        elif self.args.cipher == "aes224":
            X, Y, ctdata0l, ctdata0r, ctdata1l, ctdata1r = self.make_train_data_generalaes224(n)
        elif self.args.cipher == "aes228":
            X, Y, ctdata0l, ctdata0r, ctdata1l, ctdata1r = self.make_train_data_generalaes228(n)
        return (X, Y, ctdata0l, ctdata0r, ctdata1l, ctdata1r);


    def make_train_data_general(self, n):
        Y = np.frombuffer(self.urandom_from_random(n), dtype=np.uint8);
        Y = Y & 1;
        keys = np.frombuffer(self.urandom_from_random(8 * n), dtype=np.uint16).reshape(4, -1);
        plain0l = np.frombuffer(self.urandom_from_random(2 * n), dtype=np.uint16);
        plain0r = np.frombuffer(self.urandom_from_random(2 * n), dtype=np.uint16);
        plain1l = plain0l ^ self.diff[0];
        plain1r = plain0r ^ self.diff[1];
        num_rand_samples = np.sum(Y == 0);
        if self.args.type_create_data == "normal":
            plain1l[Y == 0] = np.frombuffer(self.urandom_from_random( 2 * num_rand_samples), dtype=np.uint16);
            plain1r[Y == 0] = np.frombuffer(self.urandom_from_random(2 * num_rand_samples), dtype=np.uint16);
        ks = self.cipher.expand_key(keys, self.args.nombre_round_eval);
        ctdata0l, ctdata0r = self.cipher.encrypt((plain0l, plain0r), ks);
        ctdata1l, ctdata1r = self.cipher.encrypt((plain1l, plain1r), ks);
        if self.args.type_create_data == "real_difference":
            k0 = np.frombuffer(self.urandom_from_random(2 * num_rand_samples), dtype=np.uint16);
            k1 = np.frombuffer(self.urandom_from_random(2 * num_rand_samples), dtype=np.uint16);
            ctdata0l[Y == 0] = ctdata0l[Y == 0] ^ k0;
            ctdata0r[Y == 0] = ctdata0r[Y == 0] ^ k1;
            ctdata1l[Y == 0] = ctdata1l[Y == 0] ^ k0;
            ctdata1r[Y == 0] = ctdata1r[Y == 0] ^ k1;
        liste_inputs = self.convert_data_inputs(self.args, ctdata0l, ctdata0r, ctdata1l, ctdata1r)
        X = self.convert_to_binary(liste_inputs);
        return (X, Y, ctdata0l, ctdata0r, ctdata1l, ctdata1r);




    def make_train_data_general_8class(self, n):
        Y  = np.random.choice(np.arange(1, len(self.diffs)+1), n, p=self.ps)
        #Y[Y != 0] = 1
        keys = np.frombuffer(self.urandom_from_random(8 * n), dtype=np.uint16).reshape(4, -1);
        plain0l = np.frombuffer(self.urandom_from_random(2 * n), dtype=np.uint16);
        plain0r = np.frombuffer(self.urandom_from_random(2 * n), dtype=np.uint16);
        plain1l = np.frombuffer(self.urandom_from_random(2 * n), dtype=np.uint16).copy();
        plain1r = np.frombuffer(self.urandom_from_random(2 * n), dtype=np.uint16).copy();
        for classe in range(len(self.diffs)):
            plain1l[Y == classe+1] = plain0l[Y == classe+1] ^ self.diffs[classe][0];
            plain1r[Y == classe+1] = plain0r[Y == classe+1] ^ self.diffs[classe][1];
        Y[Y == 8] = 4
        Y[Y == 7] = 4
        Y[Y == 6] = 4
        Y[Y == 5] = 4
        ks = self.cipher.expand_key(keys, self.args.nombre_round_eval);
        ctdata0l, ctdata0r = self.cipher.encrypt((plain0l, plain0r), ks);
        ctdata1l, ctdata1r = self.cipher.encrypt((plain1l, plain1r), ks);

        Y2 = np.zeros(int(n/8), dtype=np.uint16)
        nombre_round_eval2 = self.args.nombre_round_eval +2
        keys = np.frombuffer(self.urandom_from_random(8 * len(Y2)), dtype=np.uint16).reshape(4, -1);
        plain0l = np.frombuffer(self.urandom_from_random(2 * len(Y2)), dtype=np.uint16);
        plain0r = np.frombuffer(self.urandom_from_random(2 * len(Y2)), dtype=np.uint16);
        plain1l = np.frombuffer(self.urandom_from_random( 2 * len(Y2)), dtype=np.uint16);
        plain1r = np.frombuffer(self.urandom_from_random(2 * len(Y2)), dtype=np.uint16);
        ks = self.cipher.expand_key(keys, nombre_round_eval2);
        ctdata0l2, ctdata0r2 = self.cipher.encrypt((plain0l, plain0r), ks);
        ctdata1l2, ctdata1r2 = self.cipher.encrypt((plain1l, plain1r), ks);

        Y = np.append(Y, Y2)
        ctdata0l = np.append(ctdata0l, ctdata0l2)
        ctdata0r = np.append(ctdata0r, ctdata0r2)
        ctdata1l = np.append(ctdata1l, ctdata1l2)
        ctdata1r = np.append(ctdata1r, ctdata1r2)

        liste_inputs = self.convert_data_inputs(self.args, ctdata0l, ctdata0r, ctdata1l, ctdata1r)
        X = self.convert_to_binary(liste_inputs);
        return (X, Y, ctdata0l, ctdata0r, ctdata1l, ctdata1r);


    def make_train_data_N_batch(self, n, Nbatch):

        keys = np.frombuffer(self.urandom_from_random(8 * n), dtype=np.uint16).reshape(4, -1);
        ks = self.cipher.expand_key(keys, self.args.nombre_round_eval);
        Y = np.frombuffer(self.urandom_from_random(n), dtype=np.uint8);
        Y = Y & 1;


        for batch in range(Nbatch):
            plain0l = np.frombuffer(self.urandom_from_random(2 * n), dtype=np.uint16);
            plain0r = np.frombuffer(self.urandom_from_random(2 * n), dtype=np.uint16);
            plain1l = plain0l ^ self.diff[0];
            plain1r = plain0r ^ self.diff[1];
            num_rand_samples = np.sum(Y == 0);
            if self.args.type_create_data == "normal":
                plain1l[Y == 0] = np.frombuffer(self.urandom_from_random( 2 * num_rand_samples), dtype=np.uint16);
                plain1r[Y == 0] = np.frombuffer(self.urandom_from_random(2 * num_rand_samples), dtype=np.uint16);
            ctdata0l, ctdata0r = self.cipher.encrypt((plain0l, plain0r), ks);
            ctdata1l, ctdata1r = self.cipher.encrypt((plain1l, plain1r), ks);
            liste_inputs = self.convert_data_inputs(self.args, ctdata0l, ctdata0r, ctdata1l, ctdata1r)
            X = self.convert_to_binary(liste_inputs);
            if batch==0:
                Xfinal = X
            else:
                Xfinal = np.concatenate((Xfinal, X), axis=1)


        #print(Xfinal.shape)
        return (Xfinal, Y, ctdata0l, ctdata0r, ctdata1l, ctdata1r);


    def make_train_data_general_3class(self, n):

        keys = np.frombuffer(self.urandom_from_random(8 * n), dtype=np.uint16).reshape(4, -1);
        Y = np.frombuffer(self.urandom_from_random(n), dtype=np.uint8);
        Y = Y % 3;
        plain0l = np.frombuffer(self.urandom_from_random(2 * n), dtype=np.uint16);
        plain0r = np.frombuffer(self.urandom_from_random(2 * n), dtype=np.uint16);
        plain1l = plain0l ^ self.diff[0];
        plain1r = plain0r ^ self.diff[1];
        ind0 = np.where(Y == 0)
        num0 = np.sum(Y == 0)
        num1 = np.sum(Y == 1)
        num2 = np.sum(Y == 2)

        print(num0, num1, num2)

        ks = self.cipher.expand_key(keys, 4);
        ks2 = self.cipher.expand_key(keys, self.args.nombre_round_eval);
        ctdata0l, ctdata0r = self.cipher.encrypt((plain0l, plain0r), ks);
        ctdata1l, ctdata1r = self.cipher.encrypt((plain1l, plain1r), ks);
        c0l, c0r = self.cipher.encrypt((plain0l, plain0r), ks2);
        c1l, c1r = self.cipher.encrypt((plain1l, plain1r), ks2);
        dL = ctdata0l ^ ctdata1l
        dR = ctdata0r ^ ctdata1r
        m = 0b1100000111000011
        valL = 0b1000000100000010
        valR = 0b1000000100000000
        ind1 = np.where(np.logical_and((dL & m) == valL, (dR & m) == valR))
        ind2 = np.where(np.logical_or((dL & m) != valL, (dR & m) != valR))

        Xl01 = c0l[ind1][range(num1)]
        Xr01 = c0r[ind1][range(num1)]
        Xl11 = c1l[ind1][range(num1)]
        Xr11 = c1r[ind1][range(num1)]

        Xl02 = c0l[ind2][range(num2)]
        Xr02 = c0r[ind2][range(num2)]
        Xl12 = c1l[ind2][range(num2)]
        Xr12 = c1r[ind2][range(num2)]

        ctdata0l[Y == 0] = np.frombuffer(self.urandom_from_random(2 * num0), dtype=np.uint16)
        ctdata0r[Y == 0] = np.frombuffer(self.urandom_from_random(2 * num0), dtype=np.uint16)
        ctdata1l[Y == 0] = np.frombuffer(self.urandom_from_random(2 * num0), dtype=np.uint16)
        ctdata1r[Y == 0] = np.frombuffer(self.urandom_from_random(2 * num0), dtype=np.uint16)

        ctdata0l[Y == 1] = Xl01
        ctdata0r[Y == 1] = Xr01
        ctdata1l[Y == 1] = Xl11
        ctdata1r[Y == 1] = Xr11

        ctdata0l[Y == 2] = Xl02
        ctdata0r[Y == 2] = Xr02
        ctdata1l[Y == 2] = Xl12
        ctdata1r[Y == 2] = Xr12

        liste_inputs = self.convert_data_inputs(self.args, ctdata0l, ctdata0r, ctdata1l, ctdata1r)
        X = self.convert_to_binary(liste_inputs);
        return (X, Y, ctdata0l, ctdata0r, ctdata1l, ctdata1r);

    def make_train_data_general_3class_v2(self, n, diff=(0x0040, 0)):
        keys = np.frombuffer(self.urandom_from_random(8 * n), dtype=np.uint16).reshape(4, -1);
        Y = np.frombuffer(self.urandom_from_random(n), dtype=np.uint8);
        Y = Y & 1;
        plain0l = np.frombuffer(self.urandom_from_random(2 * n), dtype=np.uint16);
        plain0r = np.frombuffer(self.urandom_from_random(2 * n), dtype=np.uint16);
        plain1l = plain0l ^ diff[0];
        plain1r = plain0r ^ diff[1];
        ind0 = np.where(Y == 0)
        num0 = np.sum(Y == 0)
        plain1l[Y == 0] = np.frombuffer(self.urandom_from_random(2 * num0), dtype=np.uint16);
        plain1r[Y == 0] = np.frombuffer(self.urandom_from_random(2 * num0), dtype=np.uint16);

        ks = self.cipher.expand_key(keys, 4);
        ks2 = self.cipher.expand_key(keys, self.args.nombre_round_eval);
        ctdata0l, ctdata0r = self.cipher.encrypt((plain0l, plain0r), ks);
        ctdata1l, ctdata1r = self.cipher.encrypt((plain1l, plain1r), ks);
        c0l, c0r = self.cipher.encrypt((plain0l, plain0r), ks2);
        c1l, c1r = self.cipher.encrypt((plain1l, plain1r), ks2);
        dL = ctdata0l ^ ctdata1l
        dR = ctdata0r ^ ctdata1r
        m = 0b1100000111000011
        valL = 0b1000000100000010
        valR = 0b1000000100000000
        ind1 = np.where(np.logical_and((dL & m) == valL, (dR & m) == valR, Y > 0))
        ind2 = np.where(np.logical_and(np.logical_or((dL & m) != valL, (dR & m) != valR), Y > 0))

        Y[ind2] = 2
        ctdata0l = c0l
        ctdata0r = c0r
        ctdata1l = c1l
        ctdata1r = c1r
        #  print(collections.Counter((ctdata0l^ctdata1l^ctdata0r^ctdata1r)[Y==0]).most_common(10))
        #  print(collections.Counter((ctdata0l^ctdata1l^ctdata0r^ctdata1r)[Y==1]).most_common(10))
        #  print(collections.Counter((ctdata0l^ctdata1l^ctdata0r^ctdata1r)[Y==2]).most_common(10))
        num1 = np.sum(Y == 1)
        num2 = np.sum(Y == 2)
        print(num0, num1, num2)
        liste_inputs = self.convert_data_inputs(self.args, ctdata0l, ctdata0r, ctdata1l, ctdata1r)
        X = self.convert_to_binary(liste_inputs);
        return (X, Y, ctdata0l, ctdata0r, ctdata1l, ctdata1r);


    def convert_data_inputs(self, args, ctdata0l, ctdata0r, ctdata1l, ctdata1r):
        inputs_toput = []
        if self.args.cipher =="speck":
            V0 = self.cipher.ror(ctdata0l ^ ctdata0r, self.cipher.BETA)
            V1 = self.cipher.ror(ctdata1l ^ ctdata1r, self.cipher.BETA)
            DV = V0 ^ V1
            V0Inv = 65535 - V0 #(V0 ^ 0xffff)
            V1Inv = 65535 - V1
            inv_DeltaV = 65535 - DV
        if self.args.cipher =="simon":
            c0r =ctdata0r
            c0l = ctdata0l
            c1r = ctdata1r
            c1l = ctdata1l
            t0 = (self.cipher.rol(c0r, 8) & self.cipher.rol(c0r, 1)) ^ self.cipher.rol(c0r, 2) ^ c0l
            t1 = (self.cipher.rol(c1r, 8) & self.cipher.rol(c1r, 1)) ^ self.cipher.rol(c1r, 2) ^ c1l



        for i in range(len(args.inputs_type)):
            if args.inputs_type[i] =="ctdata0l":
                inputs_toput.append(ctdata0l)
            if args.inputs_type[i] =="ctdata1l":
                inputs_toput.append(ctdata1l)
            if args.inputs_type[i] =="ctdata0r":
                inputs_toput.append(ctdata0r)
            if args.inputs_type[i] =="ctdata1r":
                inputs_toput.append(ctdata1r)
            if args.inputs_type[i] =="V0&V1":
                inputs_toput.append(V0&V1)
            if args.inputs_type[i] =="V0|V1":
                inputs_toput.append(V0 | V1)
            if args.inputs_type[i] =="ctdata0l^ctdata1l":
                inputs_toput.append(ctdata0l^ctdata1l)
            if args.inputs_type[i] =="ctdata0l^ctdata0r":
                inputs_toput.append(ctdata0l^ctdata0r)
            if args.inputs_type[i] =="ctdata0r^ctdata1r":
                inputs_toput.append(ctdata0r^ctdata1r)
            if args.inputs_type[i] =="ctdata1l^ctdata1r":
                inputs_toput.append(ctdata1l^ctdata1r)
            if args.inputs_type[i] =="ctdata1l^ctdata0r":
                inputs_toput.append(ctdata1l^ctdata0r)
            if args.inputs_type[i] =="ctdata1r^ctdata0l":
                inputs_toput.append(ctdata1r^ctdata0l)
            if args.inputs_type[i] =="ctdata0l^ctdata1l^ctdata0r^ctdata1r":
                inputs_toput.append(ctdata0l^ctdata1l^ctdata0r^ctdata1r)
            if args.inputs_type[i] =="ctdata0r^ctdata1r^ctdata0l^ctdata1l":
                inputs_toput.append(ctdata0r^ctdata1r^ctdata0l^ctdata1l)
            if args.inputs_type[i] =="inv(V0)&V1":
                inputs_toput.append(V1&V0Inv)
            if args.inputs_type[i] =="V0&inv(V1)":
                inputs_toput.append(V0&V1Inv)
            if args.inputs_type[i] =="inv(V0)&inv(V1)":
                inputs_toput.append(V0Inv&V1Inv)
            if args.inputs_type[i] =="inv(DeltaL)":
                inv_DeltaL = 65535 - ctdata0l ^ ctdata1l
                inputs_toput.append(inv_DeltaL)
            if args.inputs_type[i] =="inv(DeltaV)":
                inv_DeltaV = 65535 - ctdata0l^ctdata1l^ctdata0r^ctdata1r
                inputs_toput.append(inv_DeltaV)
            if args.inputs_type[i] == "DeltaL&DeltaV":
                DeltaL = ctdata0l ^ ctdata1l
                inputs_toput.append(DeltaL&DV)
            if args.inputs_type[i] == "DLi":
                DeltaL = ctdata0l ^ ctdata1l
                inputs_toput.append(DeltaL)
            if args.inputs_type[i] == "DLi-1":
                DeltaL = ctdata0l ^ ctdata1l
                inputs_toput.append(DeltaL>>1)
            if args.inputs_type[i] == "DLi+1":
                DeltaL = ctdata0l ^ ctdata1l
                inputs_toput.append(DeltaL<<1)
            if args.inputs_type[i] == "DVi":
                inputs_toput.append(DV)
            if args.inputs_type[i] == "DVi-1":
                inputs_toput.append(DV>>1)
            if args.inputs_type[i] == "DVi+1":
                inputs_toput.append(DV<<1)
            if args.inputs_type[i] == "V0i":
                inputs_toput.append(V0)
            if args.inputs_type[i] == "V0i-1":
                #V0 = ctdata0l ^ ctdata0r
                inputs_toput.append(V0>>1)
            if args.inputs_type[i] == "V0i+1":
                #V0 = ctdata0l ^ ctdata0r
                inputs_toput.append(V0<<1)
            if args.inputs_type[i] == "V1i":
                #V1 = ctdata1l ^ ctdata1r
                inputs_toput.append(V1)
            if args.inputs_type[i] == "V1i-1":
                #V1 = ctdata1l ^ ctdata1r
                inputs_toput.append(V1>>1)
            if args.inputs_type[i] == "V1i+1":
                #V1 = ctdata1l ^ ctdata1r
                inputs_toput.append(V1<<1)
            if args.inputs_type[i] == "DL":
                DeltaL = ctdata0l ^ ctdata1l
                inputs_toput.append(DeltaL)
            if args.inputs_type[i] == "inv(DL)":
                DeltaL = 65535 - (ctdata0l ^ ctdata1l)
                inputs_toput.append(DeltaL)
            if args.inputs_type[i] == "V0":
                #V0 = ctdata0l ^ ctdata0r
                inputs_toput.append(V0)
            if args.inputs_type[i] == "inv(V0)":
                #V0 = 65535 - (ctdata0l ^ ctdata0r)
                inputs_toput.append(V0Inv)
            if args.inputs_type[i] == "V1":
                #V1 = ctdata1l ^ ctdata1r
                inputs_toput.append(V1)
            if args.inputs_type[i] == "inv(V1)":
                #V1 = 65535 - (ctdata1l ^ ctdata1r)
                inputs_toput.append(V1Inv)
            if args.inputs_type[i] == "DV":
                #V1 = ctdata1l ^ ctdata1r
                inputs_toput.append(DV)
            if args.inputs_type[i] == "inv(DV)":
                #V1 = 65535 - (ctdata1l ^ ctdata1r)
                inputs_toput.append(inv_DeltaV)
            if args.inputs_type[i] == "c0r^c1r":
                #V1 = 65535 - (ctdata1l ^ ctdata1r)
                inputs_toput.append(c0r^c1r)
            if args.inputs_type[i] == "c0l^c1l":
                #V1 = 65535 - (ctdata1l ^ ctdata1r)
                inputs_toput.append(c0l^c1l)
            if args.inputs_type[i] == "t0^t1":
                #V1 = 65535 - (ctdata1l ^ ctdata1r)
                inputs_toput.append(t0^t1)

        return inputs_toput


    def find_difference(self, c0l, c0r, c1l, c1r):
        # takes in n round ciphertext and output difference at n-1 round
        t0 = (rol(c0r, 8) & rol(c0r, 1)) ^ rol(c0r, 2) ^ c0l
        t1 = (rol(c1r, 8) & rol(c1r, 1)) ^ rol(c1r, 2) ^ c1l
        return (c0r ^ c1r, t0 ^ t1)
