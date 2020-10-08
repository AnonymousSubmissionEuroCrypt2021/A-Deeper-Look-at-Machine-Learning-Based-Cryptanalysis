import numpy as np


def urandom_from_random(rng, length):
    if length == 0:
        return b''

    import sys
    chunk_size = 65535
    chunks = []
    while length >= chunk_size:
        chunks.append(rng.getrandbits(
                chunk_size * 8).to_bytes(chunk_size, sys.byteorder))
        length -= chunk_size
    if length:
        chunks.append(rng.getrandbits(
                length * 8).to_bytes(length, sys.byteorder))
    result = b''.join(chunks)
    return result


def WORD_SIZE():
    return(8);

MASK_VAL = 2 ** WORD_SIZE() - 1;
SBox=np.array([99, 124, 119, 123, 242, 107, 111, 197,  48,   1, 103,  43, 254, 215, 171, 118, 		202, 130, 201, 125, 250,  89,  71, 240, 173, 212, 162, 175, 156, 164, 114, 192, 		183, 253, 147,  38,  54,  63, 247, 204,  52, 165, 229, 241, 113, 216,  49,  21, 		4, 199,  35, 195,  24, 150,   5, 154,   7,  18, 128, 226, 235,  39, 178, 117, 		9, 131,  44,  26,  27, 110,  90, 160,  82,  59, 214, 179,  41, 227,  47, 132, 		83, 209,   0, 237,  32, 252, 177,  91, 106, 203, 190,  57,  74,  76,  88, 207, 		208, 239, 170, 251,  67,  77,  51, 133,  69, 249,   2, 127,  80,  60, 159, 168, 		81, 163,  64, 143, 146, 157,  56, 245, 188, 182, 218,  33,  16, 255, 243, 210, 		205,  12,  19, 236,  95, 151,  68,  23, 196, 167, 126,  61, 100,  93,  25, 115, 		96, 129,  79, 220,  34,  42, 144, 136,  70, 238, 184,  20, 222,  94,  11, 219, 		224,  50,  58,  10,  73,   6,  36,  92, 194, 211, 172,  98, 145, 149, 228, 121, 		231, 200,  55, 109, 141, 213,  78, 169, 108,  86, 244, 234, 101, 122, 174,   8, 		186, 120,  37,  46,  28, 166, 180, 198, 232, 221, 116,  31,  75, 189, 139, 138, 		112,  62, 181, 102,  72,   3, 246,  14,  97,  53,  87, 185, 134, 193,  29, 158, 		225, 248, 152,  17, 105, 217, 142, 148, 155,  30, 135, 233, 206,  85,  40, 223, 		140, 161, 137,  13, 191, 230,  66, 104,  65, 153,  45,  15, 176,  84, 187,  22], dtype=np.uint8)

Mul2=np.array([
    0x00,0x02,0x04,0x06,0x08,0x0a,0x0c,0x0e,0x10,0x12,0x14,0x16,0x18,0x1a,0x1c,0x1e,    0x20,0x22,0x24,0x26,0x28,0x2a,0x2c,0x2e,0x30,0x32,0x34,0x36,0x38,0x3a,0x3c,0x3e,    0x40,0x42,0x44,0x46,0x48,0x4a,0x4c,0x4e,0x50,0x52,0x54,0x56,0x58,0x5a,0x5c,0x5e,    0x60,0x62,0x64,0x66,0x68,0x6a,0x6c,0x6e,0x70,0x72,0x74,0x76,0x78,0x7a,0x7c,0x7e,    0x80,0x82,0x84,0x86,0x88,0x8a,0x8c,0x8e,0x90,0x92,0x94,0x96,0x98,0x9a,0x9c,0x9e,    0xa0,0xa2,0xa4,0xa6,0xa8,0xaa,0xac,0xae,0xb0,0xb2,0xb4,0xb6,0xb8,0xba,0xbc,0xbe,    0xc0,0xc2,0xc4,0xc6,0xc8,0xca,0xcc,0xce,0xd0,0xd2,0xd4,0xd6,0xd8,0xda,0xdc,0xde,    0xe0,0xe2,0xe4,0xe6,0xe8,0xea,0xec,0xee,0xf0,0xf2,0xf4,0xf6,0xf8,0xfa,0xfc,0xfe,    0x1b,0x19,0x1f,0x1d,0x13,0x11,0x17,0x15,0x0b,0x09,0x0f,0x0d,0x03,0x01,0x07,0x05,    0x3b,0x39,0x3f,0x3d,0x33,0x31,0x37,0x35,0x2b,0x29,0x2f,0x2d,0x23,0x21,0x27,0x25,    0x5b,0x59,0x5f,0x5d,0x53,0x51,0x57,0x55,0x4b,0x49,0x4f,0x4d,0x43,0x41,0x47,0x45,    0x7b,0x79,0x7f,0x7d,0x73,0x71,0x77,0x75,0x6b,0x69,0x6f,0x6d,0x63,0x61,0x67,0x65,    0x9b,0x99,0x9f,0x9d,0x93,0x91,0x97,0x95,0x8b,0x89,0x8f,0x8d,0x83,0x81,0x87,0x85,    0xbb,0xb9,0xbf,0xbd,0xb3,0xb1,0xb7,0xb5,0xab,0xa9,0xaf,0xad,0xa3,0xa1,0xa7,0xa5,    0xdb,0xd9,0xdf,0xdd,0xd3,0xd1,0xd7,0xd5,0xcb,0xc9,0xcf,0xcd,0xc3,0xc1,0xc7,0xc5,    0xfb,0xf9,0xff,0xfd,0xf3,0xf1,0xf7,0xf5,0xeb,0xe9,0xef,0xed,0xe3,0xe1,0xe7,0xe5], dtype=np.uint8);

Rcon=np.array([
0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36, 0x6c, 0xd8, 0xab, 0x4d, 0x9a, 0x2f, 0x5e, 0xbc, 0x63, 0xc6, 0x97, 0x35, 0x6a, 0xd4, 0xb3, 0x7d, 0xfa, 0xef, 0xc5, 0x91, 0x39, 0x72, 0xe4, 0xd3, 0xbd, 0x61, 0xc2, 0x9f, 0x25, 0x4a, 0x94, 0x33, 0x66, 0xcc, 0x83, 0x1d, 0x3a, 0x74, 0xe8, 0xcb, 0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36, 0x6c, 0xd8, 0xab, 0x4d, 0x9a, 0x2f, 0x5e, 0xbc, 0x63, 0xc6, 0x97, 0x35, 0x6a, 0xd4, 0xb3, 0x7d, 0xfa, 0xef, 0xc5, 0x91, 0x39, 0x72, 0xe4, 0xd3, 0xbd, 0x61, 0xc2, 0x9f, 0x25, 0x4a, 0x94, 0x33, 0x66, 0xcc, 0x83, 0x1d, 0x3a, 0x74, 0xe8, 0xcb, 0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36, 0x6c, 0xd8, 0xab, 0x4d, 0x9a, 0x2f, 0x5e, 0xbc, 0x63, 0xc6, 0x97, 0x35, 0x6a, 0xd4, 0xb3, 0x7d, 0xfa, 0xef, 0xc5, 0x91, 0x39, 0x72, 0xe4, 0xd3, 0xbd, 0x61, 0xc2, 0x9f, 0x25, 0x4a, 0x94, 0x33, 0x66, 0xcc, 0x83, 0x1d, 0x3a, 0x74, 0xe8, 0xcb, 0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36, 0x6c, 0xd8, 0xab, 0x4d, 0x9a, 0x2f, 0x5e, 0xbc, 0x63, 0xc6, 0x97, 0x35, 0x6a, 0xd4, 0xb3, 0x7d, 0xfa, 0xef, 0xc5, 0x91, 0x39, 0x72, 0xe4, 0xd3, 0xbd, 0x61, 0xc2, 0x9f, 0x25, 0x4a, 0x94, 0x33, 0x66, 0xcc, 0x83, 0x1d, 0x3a, 0x74, 0xe8, 0xcb, 0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36, 0x6c, 0xd8, 0xab, 0x4d, 0x9a, 0x2f, 0x5e, 0xbc, 0x63, 0xc6, 0x97, 0x35, 0x6a, 0xd4, 0xb3, 0x7d, 0xfa, 0xef, 0xc5, 0x91, 0x39, 0x72, 0xe4, 0xd3, 0xbd, 0x61, 0xc2, 0x9f, 0x25, 0x4a, 0x94, 0x33, 0x66, 0xcc, 0x83, 0x1d, 0x3a, 0x74, 0xe8, 0xcb], dtype=np.uint8)

def mul2(x):
  return Mul2[x]

def mul3(x):
  return Mul2[x]^x

def SB(p):
  return SBox[p]#[SBox[p[0]], SBox[p[1]], SBox[p[2]], SBox[p[3]]]

def SR(p):
  return np.array([p[0], p[3], p[2], p[1]], dtype=np.uint8)

def MC(p):
  # First col
  t0=mul3(p[0])^mul2(p[1])
  t1=mul2(p[0])^mul3(p[1])

  #second col
  t2=mul3(p[2])^mul2(p[3])
  t3=mul2(p[2])^mul3(p[3])
  return np.array([t0, t1, t2, t3], dtype=np.uint8)

def ARK(p, k):
  return p^k

def KS(k, i):
  tmp0=SBox[k[3]]^Rcon[i]
  tmp1=SBox[k[2]]
  t0=tmp0^k[0]
  t1=tmp1^k[1]
  t2=k[2]^t0
  t3=k[3]^t1
  return np.array([t0,t1,t2,t3], dtype=np.uint8)

def enc_one_round(p, k):
    p = SB(p)
    p = SR(p)
    p = MC(p)
    p = ARK(p,k);
    return(p);

def encrypt_AES228(p, k, nr):
    p=ARK(p,k)
    for r in range(1,nr):
        k=KS(k, r)
        p = enc_one_round(p, k);
    # last round
    p = SB(p)
    p = SR(p)
    k=KS(k, nr)
    p = ARK(p,k);
    #p = MC(p)
    return(p);

def make_train_data_AES228_v1(rng, n, nr, diff=(1,0,0,1)):
    Y = np.frombuffer(urandom_from_random(rng, n), dtype=np.uint8);
    Y = Y & 1;
    p0 = np.frombuffer(urandom_from_random(rng, 4 * n), dtype=np.uint8).reshape(4, n);
    p1 = p0 ^ (np.array(diff).reshape(4, 1))
    k = np.frombuffer(urandom_from_random(rng, 4 * n), dtype=np.uint8).reshape(4, n);
    nrand = np.sum(Y == 0)
    # p1[,Y==0]=np.frombuffer(urandom(4*nrand),dtype=np.uint8).reshape(4,-1);
    c0 = encrypt_AES228(p0, k, nr)
    c1 = encrypt_AES228(p1, k, nr)
    C0 = np.transpose(c0)
    C1 = np.transpose(c1)
    C1[Y == 0] = np.frombuffer(urandom_from_random(rng, 4 * nrand), dtype=np.uint8).reshape(nrand, -1);
    C = np.concatenate((C0, C1), axis=1)
    X = np.unpackbits(C, axis=1)
    C0_new = np.unpackbits(C0, axis=1)
    C1_new = np.unpackbits(C1, axis=1)
    return(X, Y, C0_new[:, :16], C0_new[:, 16:], C1_new[:, :16], C1_new[:, 16:]);

def make_train_data_AES228_v2(rng, n, nr, diff=(1,0,0,1)):
  Y = np.frombuffer(urandom_from_random(rng, n), dtype=np.uint8); Y = Y & 1;
  p0 = (np.frombuffer(urandom_from_random(rng,4* n),dtype=np.uint8)).reshape(4,n);
  p1 = (p0.copy())^(np.array(diff).reshape(4,1))
  k = (np.frombuffer(urandom_from_random(rng,4* n),dtype=np.uint8)).reshape(4,n);
  nrand=np.sum(Y==0)
  c0=encrypt_AES228(p0,k,nr)
  c1=encrypt_AES228(p1,k,nr)
  m=(np.frombuffer(urandom_from_random(rng,4* nrand),dtype=np.uint8)).reshape((c1[:,Y==0]).shape)
  c0[:,Y==0]^=m;
  c1[:,Y==0]^=m;
  c=np.concatenate((c0,c1), axis=0)
  X=convert_to_binary(c)
  return (X, Y, X[:, :16], X[:, 16:32], X[:, 32:48], X[:, 48:]);

def make_train_data_AES228_v3(rng, n, nr, flag_key_unique=False):
    Y = np.frombuffer(urandom_from_random(rng, n), dtype=np.uint8);
    Y = Y & 1;
    p0 = np.frombuffer(urandom_from_random(rng, 4 * n), dtype=np.uint8).reshape(4, n);
    #p1 = p0 ^ (np.array(diff).reshape(4, 1))
    if not flag_key_unique:
        k = np.frombuffer(urandom_from_random(rng, 4 * n), dtype=np.uint8).reshape(4, n);
    else:
        k = np.frombuffer(urandom_from_random(rng, 4 * n), dtype=np.uint8).copy()
        x = 97
        print(x)
        k[k>-100] = x
        k = k.reshape(4, n);
    nrand = np.sum(Y == 0)
    # p1[,Y==0]=np.frombuffer(urandom(4*nrand),dtype=np.uint8).reshape(4,-1);
    c0 = encrypt_AES228(p0, k, nr)
    #c1 = encrypt_AES228(p1, k, nr)
    C0 = np.transpose(c0)
    #C1 = np.transpose(c1)
    C0[Y == 0] = np.frombuffer(urandom_from_random(rng, 4 * nrand), dtype=np.uint8).reshape(nrand, -1);
    p0 = np.transpose(p0)
    C = np.concatenate((C0, p0), axis=1)
    X = np.unpackbits(C, axis=1)
    C0_new = np.unpackbits(C0, axis=1)
    p0 = np.unpackbits(p0, axis=1)
    return (X, Y, C0_new[:, :16], C0_new[:, 16:], p0[:, :16], p0[:, 16:]);

def convert_to_binary(arr):
  nbWords=len(arr);
  X = np.zeros((nbWords * WORD_SIZE(),len(arr[0])),dtype=np.uint8);
  for i in range(nbWords * WORD_SIZE()):
    index = i // WORD_SIZE();
    offset = WORD_SIZE() - (i % WORD_SIZE()) - 1;
    X[i] = (arr[index] >> offset) & 1;
  X = X.transpose();
  return(X);



class Cipher_Simon:

    def __init__(self, args):
        self.args = args
        self.WORD_SIZE = self.args.word_size
        self.ALPHA = self.args.alpha
        self.BETA = self.args.beta
        self.MASK_VAL = 2 ** self.WORD_SIZE - 1;

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




