import numpy as np
from os import urandom

def WORD_SIZE():
    return(4);

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

MASK_VAL = 2 ** WORD_SIZE() - 1;

SBox=np.array([0x6,0xB,0x5,0x4,0x2,0xE,0x7,0xA,0x9,0xD,0xF,0xC,0x3,0x1,0x0,0x8], dtype=np.uint8)

Mul2=np.array([0, 2, 4, 6, 8, 10, 12, 14, 3, 1, 7, 5, 11, 9, 15, 13], dtype=np.uint8);

Rcon=np.array([9, 1, 2, 4, 8, 3, 6, 12, 11, 5, 10, 7, 14, 15, 13], dtype=np.uint8)

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

def encrypt_AES224(p, k, nr):
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


def make_train_data_AES224_v1(rng, n, nr, diff=(1,0,0,1)):
  Y = np.frombuffer(urandom_from_random(rng, n), dtype=np.uint8); Y = Y & 1;
  p0 = (np.frombuffer(urandom_from_random(rng, 4*n),dtype=np.uint8)&0xf).reshape(4,n);
  p1 = (p0.copy())^(np.array(diff).reshape(4,1))
  k = (np.frombuffer(urandom_from_random(rng, 4*n),dtype=np.uint8)&0xf).reshape(4,n);
  nrand=np.sum(Y==0)
  c0=encrypt_AES224(p0,k,nr)
  c1=encrypt_AES224(p1,k,nr)
  c1[:,Y==0]=(np.frombuffer(urandom(4*nrand),dtype=np.uint8)&0xf).reshape((c1[:,Y==0]).shape);
  c=np.concatenate((c0,c1), axis=0)
  X=convert_to_binary(c)
  return (X, Y, X[:, :16], X[:, 16:32], X[:, 32:48], X[:, 48:]);


#real differences data generator
def make_train_data_AES224_v2(rng, n, nr, diff=(1,0,0,1)):
  Y = np.frombuffer(urandom_from_random(rng, n), dtype=np.uint8); Y = Y & 1;
  p0 = (np.frombuffer(urandom_from_random(rng, 4*n),dtype=np.uint8)&0xf).reshape(4,n);
  p1 = (p0.copy())^(np.array(diff).reshape(4,1))
  k = (np.frombuffer(urandom_from_random(rng, 4*n),dtype=np.uint8)&0xf).reshape(4,n);
  nrand=np.sum(Y==0)
  c0=encrypt_AES224(p0,k,nr)
  c1=encrypt_AES224(p1,k,nr)
  m=(np.frombuffer(urandom(4*nrand),dtype=np.uint8)&0xf).reshape((c1[:,Y==0]).shape)
  c0[:,Y==0]^=m;
  c1[:,Y==0]^=m;
  c=np.concatenate((c0,c1), axis=0)
  X=convert_to_binary(c)
  return (X, Y, X[:, :16], X[:, 16:32], X[:, 32:48], X[:, 48:]);

def make_train_data_AES224_v3(rng, n, nr, flag_key_unique=False):
    Y = np.frombuffer(urandom_from_random(rng, n), dtype=np.uint8);
    Y = Y & 1;
    p0 = (np.frombuffer(urandom_from_random(rng, 4 * n), dtype=np.uint8) & 0xf).reshape(4, n);
    #p1 = (p0.copy()) ^ (np.array(diff).reshape(4, 1))


    if not flag_key_unique:
        k = (np.frombuffer(urandom_from_random(rng, 4 * n), dtype=np.uint8) & 0xf).reshape(4, n);

    else:
        k = (np.frombuffer(urandom_from_random(rng, 4 * n), dtype=np.uint8) & 0xf).copy()
        x = 13
        print(x)
        k[k>-100] = x
        k = k.reshape(4, n);

    nrand = np.sum(Y == 0)
    c0 = encrypt_AES224(p0, k, nr)
    #c1 = encrypt_AES224(p1, k, nr)
    p0[:, Y == 0] = (np.frombuffer(urandom(4 * nrand), dtype=np.uint8) & 0xf).reshape((p0[:, Y == 0]).shape);
    c = np.concatenate((c0, p0), axis=0)
    X = convert_to_binary(c)
    return (X, Y, X[:, :16], X[:, 16:32], X[:, 32:48], X[:, 48:]);


def convert_to_binary(arr):
  nbWords=len(arr);
  X = np.zeros((nbWords * WORD_SIZE(),len(arr[0])),dtype=np.uint8);
  for i in range(nbWords * WORD_SIZE()):
    index = i // WORD_SIZE();
    offset = WORD_SIZE() - (i % WORD_SIZE()) - 1;
    X[i] = (arr[index] >> offset) & 1;
  X = X.transpose();
  return(X);

