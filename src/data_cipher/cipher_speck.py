class Cipher_Speck:

    def __init__(self, args):
        self.args = args
        self.WORD_SIZE = self.args.word_size
        self.ALPHA = self.args.alpha
        self.BETA = self.args.beta
        self.MASK_VAL = 2 ** self.WORD_SIZE - 1;

    def rol(self, x, k):
        return (((x << k) & self.MASK_VAL) | (x >> (self.WORD_SIZE - k)));

    def ror(self, x, k):
        return ((x >> k) | ((x << (self.WORD_SIZE - k)) & self.MASK_VAL));

    def enc_one_round(self, p, k):
        c0, c1 = p[0], p[1];
        c0 = self.ror(c0, self.ALPHA);
        c0 = (c0 + c1) & self.MASK_VAL;
        c0 = c0 ^ k;
        c1 = self.rol(c1, self.BETA);
        c1 = c1 ^ c0;
        return (c0, c1);

    def dec_one_round(self, c, k):
        c0, c1 = c[0], c[1];
        c1 = c1 ^ c0;
        c1 = self.ror(c1, self.BETA);
        c0 = c0 ^ k;
        c0 = (c0 - c1) & self.MASK_VAL;
        c0 = self.rol(c0, self.ALPHA);
        return (c0, c1);

    def expand_key(self, k, t):
        ks = [0 for i in range(t)];
        ks[0] = k[len(k) - 1];
        l = list(reversed(k[:len(k) - 1]));
        for i in range(t - 1):
            l[i % 3], ks[i + 1] = self.enc_one_round((l[i % 3], ks[i]), i);
        return (ks);

    def encrypt(self, p, ks):
        x, y = p[0], p[1];
        for k in ks:
            x, y = self.enc_one_round((x, y), k);
        return (x, y);

    def decrypt(self, c, ks):
        x, y = c[0], c[1];
        for k in reversed(ks):
            x, y = self.dec_one_round((x, y), k);
        return (x, y);


