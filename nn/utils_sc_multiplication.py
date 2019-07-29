from crypto.fe_simple import FEMultiplication
from charm.toolbox.integergroup import IntegerGroup

import numpy as np
import multiprocessing
import time

debug = False

global fe_multiplication, mpk_fem, msk_fem


def initialize_fem(secparam=256):
    fe_multiplication = FEMultiplication(
        IntegerGroup(),
        p=90841625992899044736915068676923590086910503646037290972161689497324782922043,
        q=45420812996449522368457534338461795043455251823018645486080844748662391461021
    )
    mpk_fem, msk_fem = fe_multiplication.setup(secparam=256)
    return fe_multiplication, mpk_fem, msk_fem


def secure_multiplication_enc_wrap(y, i, j):
    # print("process id %s " % os.getpid())
    ct = fe_multiplication.encrypt_with_serialize(mpk_fem, y)
    return ct, i, j


def secure_multiplication_key_wrap(commitment, a, i, j):
    # print("process id %s " % os.getpid())
    sk = fe_multiplication.keyder_with_serialize(msk_fem, commitment, a)
    return sk, i, j


def secure_multiplication_dec_wrap(ct, sk, a, i, j, max_prod):
    # print("process id %s " % os.getpid())
    dec_mul = fe_multiplication.decrypt_with_deserialize(mpk_fem, ct, sk, a, max_prod)
    return dec_mul, i, j


def secure_multiplication_cost(Y, A, ceil_Y, ceil_A):
    assert Y.shape[0] == A.shape[0]
    assert Y.shape[1] == A.shape[1]
    Y_tmp = (Y * ceil_Y).astype(int)
    A_tmp = (A * ceil_A).astype(int)

    global fe_multiplication, mpk_fem, msk_fem
    fe_multiplication, mpk_fem, msk_fem = initialize_fem()

    result = np.zeros((Y.shape[0], Y.shape[1]))
    enc_cost = 0
    key_cost = 0
    dec_cost = 0
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            # normal multiplication
            # result[i][j] = Y[i][j]*A[i][j]
            y = int(Y_tmp[i][j])
            a = int(A_tmp[i][j])
            t1 = time.clock()
            ct = fe_multiplication.encrypt_with_serialize(mpk_fem, y)
            t2 = time.clock()
            sk = fe_multiplication.keyder_with_serialize(msk_fem, ct['commitment'], a)
            t3 = time.clock()
            max_prod = y * a + 1
            dec_mul = fe_multiplication.decrypt_with_deserialize(mpk_fem, ct['ct'], sk, a, max_prod)
            t4 = time.clock()
            enc_cost += (t2 - t1) * 1000
            key_cost += (t3 - t2) * 1000
            dec_cost += (t4 - t3) * 1000
            result[i][j] = dec_mul
    return result / (ceil_A * ceil_Y), enc_cost, key_cost, dec_cost


def secure_multiplication_multiprocessing_cost(Y, A, ceil_Y, ceil_A):
    assert Y.shape[0] == A.shape[0]
    assert Y.shape[1] == A.shape[1]

    global fe_multiplication, mpk_fem, msk_fem
    fe_multiplication, mpk_fem, msk_fem = initialize_fem()

    Y_tmp = (Y * ceil_Y).astype(int)
    A_tmp = (A * ceil_A).astype(int)
    result = np.zeros((Y.shape[0], Y.shape[1]))

    result_tmp_enc = dict()
    result_tmp_key = dict()
    result_tmp_dec = dict()

    pool = multiprocessing.Pool()

    t1 = time.clock()
    for i in range(Y.shape[0]):
        result_tmp_enc[i] = dict()
        for j in range(Y.shape[1]):
            y = int(Y_tmp[i][j])
            res, idxi, idxj = pool.apply(func=secure_multiplication_enc_wrap,
                                         args=(y, i, j))
            result_tmp_enc[idxi][idxj] = res
    pool.close()
    pool.join()

    pool = multiprocessing.Pool()
    t2 = time.clock()
    for i in range(Y.shape[0]):
        result_tmp_key[i] = dict()
        for j in range(Y.shape[1]):
            a = int(A_tmp[i][j])
            res, idxi, idxj = pool.apply(func=secure_multiplication_key_wrap,
                                         args=(result_tmp_enc[i][j]['commitment'], a, i, j))
            result_tmp_key[idxi][idxj] = res
    pool.close()
    pool.join()

    pool = multiprocessing.Pool()
    t3 = time.clock()
    for i in range(Y.shape[0]):
        result_tmp_dec[i] = dict()
        for j in range(Y.shape[1]):
            y = int(Y_tmp[i][j])
            a = int(A_tmp[i][j])
            max_prod = y * a + 1
            res, idxi, idxj = pool.apply(func=secure_multiplication_dec_wrap,
                                         args=(
                                             result_tmp_enc[i][j]['ct'], result_tmp_key[i][j], a, i, j, max_prod))
            result_tmp_dec[idxi][idxj] = res
    pool.close()
    pool.join()
    t4 = time.clock()

    enc_cost = (t2 - t1) * 1000
    key_cost = (t3 - t2) * 1000
    dec_cost = (t4 - t3) * 1000

    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            result[i][j] = result_tmp_dec[i][j]
    return result / (ceil_A * ceil_Y), enc_cost, key_cost, dec_cost
