class Cipher_Simon:

    def __init__(self, args):
        self.args = args
        self.WORD_SIZE = self.args.word_size
        self.ALPHA = self.args.alpha
        self.BETA = self.args.beta
        self.MASK_VAL = 2 ** self.WORD_SIZE - 1;

    def rol(self, x, val):
        return ((x << val) % 2 ** self.WORD_SIZE) ^ (x >> (self.WORD_SIZE - val))



    def enc_one_round(self, p, k):
        x, y = p[0], p[1];

        # Generate all circular shifts
        ls_1_x = ((x >> (self.WORD_SIZE - 1)) + (x << 1)) & self.MASK_VAL
        ls_8_x = ((x >> (self.WORD_SIZE - 8)) + (x << 8)) & self.MASK_VAL
        ls_2_x = ((x >> (self.WORD_SIZE - 2)) + (x << 2)) & self.MASK_VAL

        # XOR Chain
        xor_1 = (ls_1_x & ls_8_x) ^ y
        xor_2 = xor_1 ^ ls_2_x
        new_x = k ^ xor_2

        return new_x, x

    def dec_one_round(self, c, k):
        """Complete One Inverse Feistel Round
        :param x: Upper bits of current ciphertext
        :param y: Lower bits of current ciphertext
        :param k: Round Key
        :return: Upper and Lower plaintext segments
        """
        x, y = c[0], c[1];

        # Generate all circular shifts
        ls_1_y = ((y >> (self.WORD_SIZE - 1)) + (y << 1)) & self.MASK_VAL
        ls_8_y = ((y >> (self.WORD_SIZE - 8)) + (y << 8)) & self.MASK_VAL
        ls_2_y = ((y >> (self.WORD_SIZE - 2)) + (y << 2)) & self.MASK_VAL

        # Inverse XOR Chain
        xor_1 = k ^ x
        xor_2 = xor_1 ^ ls_2_y
        new_x = (ls_1_y & ls_8_y) ^ xor_2
        return y, new_x

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


