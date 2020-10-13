import numpy as np
from os import urandom
class Cipher_Simeck:

    def __init__(self, args):
        self.args = args
        self.WORD_SIZE = 16
        self.ALPHA = 5
        self.BETA = 1
        self.MASK_VAL = 2 ** self.WORD_SIZE - 1;
        self.RC = 2 ** self.WORD_SIZE - 4



    def shuffle_together(self, l):
        state = np.random.get_state();
        for x in l:
            np.random.set_state(state);
            np.random.shuffle(x);

    def rol(self, x,k):
        return(((x << k) & self.MASK_VAL) | (x >> (self.WORD_SIZE - k)));

    def ror(self, x,k):
        return((x >> k) | ((x << (self.WORD_SIZE - k)) & self.MASK_VAL));

    def enc_one_round(self, p, k):
        # L = p[1]
        # R= (p[0] & rot(p[0], alpha) ^ p[1] ^ rot(p[0], beta) ^k
        c1=p[0]
        c0=(p[0] & self.rol(p[0], self.ALPHA)) ^ p[1] ^ self.rol(p[0], self.BETA) ^ k
        return(c0,c1);

    def dec_one_round(self, c,k):
        return self.enc_one_round([c[1], c[0]], k)

    def get_sequence(self, num_rounds):
        if num_rounds < 40:
            states = [1] * 5
        else:
            states = [1] * 6

        for i in range(num_rounds - 5):
            if num_rounds < 40:
                feedback = states[i + 2] ^ states[i]
            else:
                feedback = states[i + 1] ^ states[i]
            states.append(feedback)

        return tuple(states)


    def expand_key(self, k, t):
        ks = [0 for i in range(t)];
        ks[0] = k[len(k)-1];
        l = list(reversed(k[:len(k)-1]));
        seq=self.get_sequence(t)
        for i in range(t-1):
            l[i%3], ks[i+1] = self.enc_one_round((l[i%3], ks[i]), self.RC^seq[i]);
        print(ks)
        return(ks);

    def encrypt(self, p, ks):
        x, y = p[0], p[1];
        for k in ks:
            x,y = self.enc_one_round((x,y), k);
        return(x, y);

    def decrypt(self, c, ks):
        x, y = c[0], c[1];
        for k in reversed(ks):
            x, y = self.dec_one_round((x,y), k);
        return(x,y);

def check_testvector():
  key = (0x1918,0x1110,0x0908,0x0100)
  pt = (0x6565, 0x6877)
  ks = expand_key_simeck(key, 32)
  ct = encrypt_simeck(pt, ks)
  if (ct == (0x770d, 0x2c76)):
    print("Testvector verified.")
    return(True);
  else:
    print("Testvector not verified.")
    return(False);

