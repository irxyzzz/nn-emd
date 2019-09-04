import numpy as np
import multiprocessing
import os
import time
import logging

logger = logging.getLogger(__name__)


class Secure2PC():

    def __init__(self, crypto, vec_len, precision):
        self.crypto = crypto
        self.vec_len = vec_len
        self.pk = crypto.generate_public_key()
        self.precision = precision

    def client_execute(self, data):
        input_shape = data.shape
        data_list = (data * pow(10, self.precision)).astype(int).flatten().tolist()
        ct_data = self.crypto.encrypt(self.pk, data_list)
        return ct_data

    def server_key_request(self, weights):
        weights_list = (weights * pow(10, self.precision)).astype(int).flatten().tolist()
        sk = self.crypto.generate_private_key(weights_list)
        return sk

    def server_execute(self, sk, ct, weights):
        weights_list = (weights * pow(10, self.precision)).astype(int).flatten().tolist()
        max_value = 1000
        max_inner_prod = 100000000 # max_value * max_value * self.vec_len
        dec_prod = self.crypto.decrypt(self.pk, sk, weights_list, ct, max_inner_prod)
        if dec_prod is None:
            logger.debug('find a bad case - decryption: ')
            logger.debug('sk: \n' + str(sk))
            logger.debug('ct: \n' + str(ct))
            logger.debug('weights: \n' + str(weights))
            assert False
        return float(dec_prod)/pow(10, self.precision*2)





# def smc_inner_product(X, W):
#     '''
#     X, W is numpy array
#     :param X: n x m, n is row, features; m is column, #samples
#     :param W: l x n, l is row, #neuro; n is column, #features parameters
#     :return:
#     '''
#     assert X.shape[0] == W.shape[1]
#
#     res = np.zeros((W.shape[0], X.shape[1]))
#
#     feip, mpk_feip, msk_feip = initialize_feip(X.shape[0])
#     for i in range(W.shape[0]):
#         for j in range(X.shape[1]):
#             x = X[:, j]
#             y = W[i, :]
#             x_list = x.tolist()
#             y_list = y.tolist()
#             ct = feip.encrypt(mpk_feip, x_list)
#             sk = feip.keyder(msk_feip, y_list)
#             max_prod = int(np.max(x)) * int(np.max(y)) * x.shape[0] + 1
#             dec_product = feip.decrypt(mpk_feip, ct, sk, y_list, max_prod)
#             res[i][j] = dec_product
#             if debug:
#                 print("original inner product: " + str(sum(x * y)))
#                 print("  secure inner product: " + str(dec_product))
#     return res.astype(int)
#
#
# def smc_inner_product_cost(X, W):
#     '''
#     X, W is numpy array
#     :param X: n x m, n is row, features; m is column, #samples
#     :param W: l x n, l is row, #neuro; n is column, #features parameters
#     :return:
#     '''
#     if debug:
#         print("X:")
#         print(X)
#         print("W:")
#         print(W)
#     assert X.shape[0] == W.shape[1]
#     res = np.zeros((W.shape[0], X.shape[1]))
#     # global feip, mpk_feip, msk_feip
#     feip, mpk_feip, msk_feip = initialize_feip(X.shape[0])
#     enc_cost = 0
#     key_cost = 0
#     dec_cost = 0
#     for i in range(W.shape[0]):
#         for j in range(X.shape[1]):
#             x = X[:, j]
#             y = W[i, :]
#             x_list = x.tolist()
#             y_list = y.tolist()
#             # if debug:
#             #     print("x:" + str(x_list))
#             #     print("y:" + str(y_list))
#             #     print(type(x_list))
#             # if debug:
#             #     print("encryption (%s, %s)" % (i,j))
#             t1 = time.clock()
#             ct = feip.encrypt(mpk_feip, x_list)
#             # print(ct)
#             t2 = time.clock()
#             # if debug:
#             #     print("key der (%s, %s)" % (i,j))
#             sk = feip.keyder(msk_feip, y_list)
#             # print(sk)
#             t3 = time.clock()
#             # if debug:
#             #     print("decryption (%s, %s)" % (i,j))
#             max_prod = int(np.max(x)) * int(np.max(y)) * x.shape[0]
#             dec_product = feip.decrypt(mpk_feip, ct, sk, y_list, max_prod)
#             t4 = time.clock()
#             res[i][j] = dec_product
#             if debug:
#                 print("original inner product: " + str(sum(x * y)))
#                 print("  secure inner product: " + str(dec_product))
#             enc_cost += (t2 - t1)
#             key_cost += (t3 - t2)
#             dec_cost += (t4 - t3)
#     return res.astype(int), enc_cost * 1000, key_cost * 1000, dec_cost * 1000
#
#
# def smc_inner_product_float(X, W, precision_x=10, precision_w=10):
#     X_tmp = (X * precision_x).astype(int)
#     W_tmp = (W * precision_w).astype(int)
#     res = smc_inner_product(X_tmp, W_tmp)
#     return res / (precision_x * precision_w)
#
#
# def feip_enc_wrap(x, i, j):
#     global feip, mpk_feip
#     ct = feip.encrypt_serialize(mpk_feip, x)
#     return ct, i, j
#
#
# def feip_key_wrap(y, i, j):
#     global feip, msk_feip
#     sk = feip.keyder_serialize(msk_feip, y)
#     return sk, i, j
#
#
# def feip_dec_wrap(ct, sk, y, i, j, max_prod):
#     global feip, mpk_feip
#     dec_product = feip.decrypt_deserialize(mpk_feip, ct, sk, y, max_prod)
#     return dec_product, i, j
#
#
# def smc_inner_product_parallel(X, W):
#     assert X.shape[0] == W.shape[1]
#
#     result = np.zeros((W.shape[0], X.shape[1]))
#     global feip, mpk_feip, msk_feip
#     feip, mpk_feip, msk_feip = initialize_feip(X.shape[0])
#
#     pool = multiprocessing.Pool()
#     result_tmp_enc = dict()
#     result_tmp_key = dict()
#     result_tmp_dec = dict()
#
#     for i in range(W.shape[0]):
#         result_tmp_enc[i] = dict()
#         for j in range(X.shape[1]):
#             x = X[:, j]
#             x_list = x.tolist()
#             res, idxi, idxj = pool.apply(func=feip_enc_wrap, args=(x_list, i, j))
#             result_tmp_enc[idxi][idxj] = res
#     pool.close()
#     pool.join()
#
#     pool = multiprocessing.Pool()
#     for i in range(W.shape[0]):
#         result_tmp_key[i] = dict()
#         for j in range(X.shape[1]):
#             y = W[i, :]
#             y_list = y.tolist()
#             res, idxi, idxj = pool.apply(func=feip_key_wrap, args=(y_list, i, j))
#             result_tmp_key[idxi][idxj] = res
#     pool.close()
#     pool.join()
#
#     pool = multiprocessing.Pool()
#     for i in range(W.shape[0]):
#         result_tmp_dec[i] = dict()
#         for j in range(X.shape[1]):
#             x = X[:, j]
#             y = W[i, :]
#             y_list = y.tolist()
#             max_prod = int(np.max(x)) * int(np.max(y)) * x.shape[0]
#             res, idxi, idxj = pool.apply(func=feip_dec_wrap, args=(result_tmp_enc[i][j],
#                                                                    result_tmp_key[i][j],
#                                                                    y_list, i, j, max_prod))
#             result_tmp_dec[idxi][idxj] = res
#     pool.close()
#     pool.join()
#
#     for i in range(W.shape[0]):
#         for j in range(X.shape[1]):
#             result[i][j] = result_tmp_dec[i][j]
#     return result
#
#
# def smc_inner_product_parallel_cost(X, W):
#     assert X.shape[0] == W.shape[1]
#
#     result = np.zeros((W.shape[0], X.shape[1]))
#
#     global feip, mpk_feip, msk_feip
#     feip, mpk_feip, msk_feip = initialize_feip(X.shape[0])
#
#     pool = multiprocessing.Pool()
#     result_tmp_enc = dict()
#     result_tmp_key = dict()
#     result_tmp_dec = dict()
#
#     t1 = time.clock()
#     for i in range(W.shape[0]):
#         result_tmp_enc[i] = dict()
#         for j in range(X.shape[1]):
#             x = X[:, j]
#             x_list = x.tolist()
#             # print('%d, %d' % (i, j))
#             # print(type(x_list))
#             # print(type(i))
#             # print(type(j))
#             # sip_wrap_enc_cost(x_list, i, j)
#             res, idxi, idxj = pool.apply(func=feip_enc_wrap, args=(x_list, i, j))
#             # print(res)
#             # print('%d, %d' % (idxi, idxj))
#             result_tmp_enc[idxi][idxj] = res
#     pool.close()
#     pool.join()
#
#     # print(result_tmp_enc)
#
#     t2 = time.clock()
#     pool = multiprocessing.Pool()
#     for i in range(W.shape[0]):
#         result_tmp_key[i] = dict()
#         for j in range(X.shape[1]):
#             y = W[i, :]
#             y_list = y.tolist()
#             # sip_wrap_key_cost(y_list, i, j)
#             res, idxi, idxj = pool.apply(func=feip_key_wrap, args=(y_list, i, j))
#             result_tmp_key[idxi][idxj] = res
#     pool.close()
#     pool.join()
#
#     # print(result_tmp_key)
#
#     t3 = time.clock()
#     pool = multiprocessing.Pool()
#     for i in range(W.shape[0]):
#         result_tmp_dec[i] = dict()
#         for j in range(X.shape[1]):
#             x = X[:, j]
#             y = W[i, :]
#             y_list = y.tolist()
#             max_prod = int(np.max(x)) * int(np.max(y)) * x.shape[0]
#             res, idxi, idxj = pool.apply(func=feip_dec_wrap, args=(result_tmp_enc[i][j],
#                                                                    result_tmp_key[i][j],
#                                                                    y_list, i, j, max_prod))
#             result_tmp_dec[idxi][idxj] = res
#     pool.close()
#     pool.join()
#     t4 = time.clock()
#
#     enc_cost = (t2 - t1) * 1000
#     key_cost = (t3 - t2) * 1000
#     dec_cost = (t4 - t3) * 1000
#     for i in range(W.shape[0]):
#         for j in range(X.shape[1]):
#             result[i][j] = result_tmp_dec[i][j]
#     return result, enc_cost, key_cost, dec_cost
#
#
# def smc_inner_product_parallel_float(X, W, precision_x=10, precision_w=10):
#     X_tmp = (X * precision_x).astype(int)
#     W_tmp = (W * precision_w).astype(int)
#     res = smc_inner_product_parallel(X_tmp, W_tmp)
#     return res / (precision_x * precision_w)
#
#
# def smc_fundamental_operation(X, Y, op):
#     '''
#     calculate X op Y
#     :param x:
#     :param y:
#     :return:
#     '''
#     assert X.shape[0] == Y.shape[0]
#     assert X.shape[1] == Y.shape[1]
#
#     fefo, mpk_fefo, msk_fefo = initialize_fefo()
#     result = np.zeros((X.shape[0], X.shape[1]))
#
#     if op == fefo.OPERATION_SYMBOL_ADDITION:
#         for i in range(X.shape[0]):
#             for j in range(Y.shape[1]):
#                 x = int(X[i][j])
#                 y = int(Y[i][j])
#                 cmt, ct = fefo.encrypt_serialize(mpk_fefo, x)
#                 sk = fefo.keyder_serialize(mpk_fefo, msk_fefo, cmt, fefo.OPERATION_SYMBOL_ADDITION, y)
#                 max_prod = abs(x + y) + 1
#                 dec = fefo.decrypt_serialize(mpk_fefo, sk, ct, fefo.OPERATION_SYMBOL_ADDITION, y, max_prod)
#                 result[i][j] = dec
#         return result
#
#     elif op == fefo.OPERATION_SYMBOL_SUBTRACT:
#         for i in range(X.shape[0]):
#             for j in range(Y.shape[1]):
#                 x = int(X[i][j])
#                 y = int(Y[i][j])
#                 cmt, ct = fefo.encrypt_serialize(mpk_fefo, x)
#                 sk = fefo.keyder_serialize(mpk_fefo, msk_fefo, cmt, fefo.OPERATION_SYMBOL_SUBTRACT, y)
#                 max_prod = abs(x - y) + 1
#                 dec = fefo.decrypt_serialize(mpk_fefo, sk, ct, fefo.OPERATION_SYMBOL_SUBTRACT, y, max_prod)
#                 result[i][j] = dec
#         return result
#
#     elif op == fefo.OPERATION_SYMBOL_MULTIPLICATION:
#         for i in range(X.shape[0]):
#             for j in range(Y.shape[1]):
#                 x = int(X[i][j])
#                 y = int(Y[i][j])
#                 cmt, ct = fefo.encrypt_serialize(mpk_fefo, x)
#                 sk = fefo.keyder_serialize(mpk_fefo, msk_fefo, cmt, fefo.OPERATION_SYMBOL_MULTIPLICATION, y)
#                 max_prod = abs(x * y) + 1
#                 dec = fefo.decrypt_serialize(mpk_fefo, sk, ct, fefo.OPERATION_SYMBOL_MULTIPLICATION, y, max_prod)
#                 result[i][j] = dec
#         return result
#
#     elif op == fefo.OPERATION_SYMBOL_DIVISION:
#         for i in range(X.shape[0]):
#             for j in range(Y.shape[1]):
#                 x = int(X[i][j])
#                 y = int(Y[i][j])
#                 cmt, ct = fefo.encrypt_serialize(mpk_fefo, x)
#                 sk = fefo.keyder_serialize(mpk_fefo, msk_fefo, cmt, fefo.OPERATION_SYMBOL_DIVISION, y)
#                 max_prod = abs(int(x / y)) + 2
#                 dec = fefo.decrypt_serialize(mpk_fefo, sk, ct, fefo.OPERATION_SYMBOL_DIVISION, y, max_prod)
#                 result[i][j] = dec
#         return result
#
#
# def smc_fundamental_operation_cost(X, Y, op):
#     assert X.shape[0] == Y.shape[0]
#     assert X.shape[1] == Y.shape[1]
#
#     fefo, mpk_fefo, msk_fefo = initialize_fefo()
#     result = np.zeros((Y.shape[0], Y.shape[1]))
#
#     if op == fefo.OPERATION_SYMBOL_ADDITION:
#         enc_cost = 0
#         key_cost = 0
#         dec_cost = 0
#         for i in range(X.shape[0]):
#             for j in range(Y.shape[1]):
#                 x = int(X[i][j])
#                 y = int(Y[i][j])
#                 t1 = time.clock()
#                 cmt, ct = fefo.encrypt_serialize(mpk_fefo, x)
#                 t2 = time.clock()
#                 sk = fefo.keyder_serialize(mpk_fefo, msk_fefo, cmt, fefo.OPERATION_SYMBOL_ADDITION, y)
#                 t3 = time.clock()
#                 max_prod = abs(x + y) + 1
#                 dec = fefo.decrypt_serialize(mpk_fefo, sk, ct, fefo.OPERATION_SYMBOL_ADDITION, y, max_prod)
#                 t4 = time.clock()
#                 enc_cost += (t2 - t1) * 1000
#                 key_cost += (t3 - t2) * 1000
#                 dec_cost += (t4 - t3) * 1000
#                 result[i][j] = dec
#         return result, enc_cost, key_cost, dec_cost
#
#     elif op == fefo.OPERATION_SYMBOL_SUBTRACT:
#         enc_cost = 0
#         key_cost = 0
#         dec_cost = 0
#         for i in range(X.shape[0]):
#             for j in range(Y.shape[1]):
#                 x = int(X[i][j])
#                 y = int(Y[i][j])
#                 t1 = time.clock()
#                 cmt, ct = fefo.encrypt_serialize(mpk_fefo, x)
#                 t2 = time.clock()
#                 sk = fefo.keyder_serialize(mpk_fefo, msk_fefo, cmt, fefo.OPERATION_SYMBOL_SUBTRACT, y)
#                 t3 = time.clock()
#                 max_prod = abs(x - y) + 1
#                 dec = fefo.decrypt_serialize(mpk_fefo, sk, ct, fefo.OPERATION_SYMBOL_SUBTRACT, y, max_prod)
#                 t4 = time.clock()
#                 enc_cost += (t2 - t1) * 1000
#                 key_cost += (t3 - t2) * 1000
#                 dec_cost += (t4 - t3) * 1000
#                 result[i][j] = dec
#         return result, enc_cost, key_cost, dec_cost
#
#     elif op == fefo.OPERATION_SYMBOL_MULTIPLICATION:
#         enc_cost = 0
#         key_cost = 0
#         dec_cost = 0
#         for i in range(X.shape[0]):
#             for j in range(Y.shape[1]):
#                 x = int(X[i][j])
#                 y = int(Y[i][j])
#                 t1 = time.clock()
#                 cmt, ct = fefo.encrypt_serialize(mpk_fefo, x)
#                 t2 = time.clock()
#                 sk = fefo.keyder_serialize(mpk_fefo, msk_fefo, cmt, fefo.OPERATION_SYMBOL_MULTIPLICATION, y)
#                 t3 = time.clock()
#                 max_prod = abs(x * y) + 1
#                 dec = fefo.decrypt_serialize(mpk_fefo, sk, ct, fefo.OPERATION_SYMBOL_MULTIPLICATION, y, max_prod)
#                 t4 = time.clock()
#                 enc_cost += (t2 - t1) * 1000
#                 key_cost += (t3 - t2) * 1000
#                 dec_cost += (t4 - t3) * 1000
#                 result[i][j] = dec
#         return result, enc_cost, key_cost, dec_cost
#
#     elif op == fefo.OPERATION_SYMBOL_DIVISION:
#         enc_cost = 0
#         key_cost = 0
#         dec_cost = 0
#         for i in range(X.shape[0]):
#             for j in range(Y.shape[1]):
#                 x = int(X[i][j])
#                 y = int(Y[i][j])
#                 t1 = time.clock()
#                 cmt, ct = fefo.encrypt_serialize(mpk_fefo, x)
#                 t2 = time.clock()
#                 sk = fefo.keyder_serialize(mpk_fefo, msk_fefo, cmt, fefo.OPERATION_SYMBOL_DIVISION, y)
#                 t3 = time.clock()
#                 max_prod = abs(int(x / y)) + 2
#                 dec = fefo.decrypt_serialize(mpk_fefo, sk, ct, fefo.OPERATION_SYMBOL_DIVISION, y, max_prod)
#                 t4 = time.clock()
#                 enc_cost += (t2 - t1) * 1000
#                 key_cost += (t3 - t2) * 1000
#                 dec_cost += (t4 - t3) * 1000
#                 result[i][j] = dec
#         return result, enc_cost, key_cost, dec_cost
#
#
# def smc_fundamental_operation_float(X, Y, op, precision_x=1000, precision_y=1000):
#     X_tmp = (X * precision_x).astype(int)
#     Y_tmp = (Y * precision_y).astype(int)
#     res = smc_fundamental_operation(X_tmp, Y_tmp, op)
#     if op == 'addition' or op == 'subtract':
#         return res / precision_x
#     elif op == 'multiplication':
#         return res / (precision_x * precision_y)
#     else:
#         return res
#
#
# def fefo_enc_wrap(x, i, j):
#     # print("process id %s " % os.getpid())
#     global fefo, mpk_fefo
#     cmt, ct = fefo.encrypt_serialize(mpk_fefo, x)
#     return cmt, ct, i, j
#
#
# def fefo_key_wrap(cmt, op, y, i, j):
#     # print("process id %s " % os.getpid())
#     global fefo, mpk_fefo, msk_fefo
#     sk = fefo.keyder_serialize(mpk_fefo, msk_fefo, cmt, op, y)
#     return sk, i, j
#
#
# def fefo_dec_wrap(ct, sk, op, y, i, j, max_dlog):
#     # print("process id %s " % os.getpid())
#     global fefo, mpk_fefo
#     dec = fefo.decrypt_serialize(mpk_fefo, sk, ct, op, y, max_dlog)
#     return dec, i, j
#
#
# def smc_fundamental_operation_parallel(X, Y, op):
#     assert X.shape[0] == Y.shape[0]
#     assert X.shape[1] == Y.shape[1]
#
#     global fefo, mpk_fefo, msk_fefo
#     fefo, mpk_fefo, msk_fefo = initialize_fefo()
#
#     result = np.zeros((Y.shape[0], Y.shape[1]))
#
#     result_tmp_enc_ct = dict()
#     result_tmp_enc_cmt = dict()
#     result_tmp_key = dict()
#     result_tmp_dec = dict()
#
#     pool = multiprocessing.Pool()
#
#     for i in range(Y.shape[0]):
#         result_tmp_enc_ct[i] = dict()
#         result_tmp_enc_cmt[i] = dict()
#         for j in range(Y.shape[1]):
#             x = int(X[i][j])
#             cmt, ct, idxi, idxj = pool.apply(func=fefo_enc_wrap,
#                                              args=(x, i, j))
#             result_tmp_enc_cmt[idxi][idxj] = cmt
#             result_tmp_enc_ct[idxi][idxj] = ct
#     pool.close()
#     pool.join()
#
#     # print(result_tmp_enc_cmt)
#     # print(result_tmp_enc_ct)
#
#     pool = multiprocessing.Pool()
#     for i in range(Y.shape[0]):
#         result_tmp_key[i] = dict()
#         for j in range(Y.shape[1]):
#             y = int(Y[i][j])
#             res, idxi, idxj = pool.apply(func=fefo_key_wrap,
#                                          args=(result_tmp_enc_cmt[i][j], op, y, i, j))
#             result_tmp_key[idxi][idxj] = res
#     pool.close()
#     pool.join()
#
#     # print(result_tmp_key)
#
#     pool = multiprocessing.Pool()
#     for i in range(Y.shape[0]):
#         result_tmp_dec[i] = dict()
#         for j in range(Y.shape[1]):
#             x = int(X[i][j])
#             y = int(Y[i][j])
#             if op == fefo.OPERATION_SYMBOL_ADDITION:
#                 max_prod = abs(x + y) + 1
#             elif op == fefo.OPERATION_SYMBOL_SUBTRACT:
#                 max_prod = abs(x - y) + 1
#             elif op == fefo.OPERATION_SYMBOL_MULTIPLICATION:
#                 max_prod = abs(x * y) + 1
#             elif op == fefo.OPERATION_SYMBOL_DIVISION:
#                 max_prod = int(abs(x / y)) + 1
#             # print(max_prod)
#             res, idxi, idxj = pool.apply(func=fefo_dec_wrap,
#                                          args=(
#                                              result_tmp_enc_ct[i][j],
#                                              result_tmp_key[i][j], op, y, i, j, max_prod))
#             result_tmp_dec[idxi][idxj] = res
#     pool.close()
#     pool.join()
#
#     # print(result_tmp_dec)
#
#     for i in range(Y.shape[0]):
#         for j in range(Y.shape[1]):
#             result[i][j] = result_tmp_dec[i][j]
#     return result
#
#
# def smc_fundamental_operation_parallel_cost(X, Y, op):
#     assert X.shape[0] == Y.shape[0]
#     assert X.shape[1] == Y.shape[1]
#
#     global fefo, mpk_fefo, msk_fefo
#     fefo, mpk_fefo, msk_fefo = initialize_fefo()
#
#     result = np.zeros((Y.shape[0], Y.shape[1]))
#
#     result_tmp_enc_ct = dict()
#     result_tmp_enc_cmt = dict()
#     result_tmp_key = dict()
#     result_tmp_dec = dict()
#
#     pool = multiprocessing.Pool()
#
#     t1 = time.clock()
#     for i in range(Y.shape[0]):
#         result_tmp_enc_ct[i] = dict()
#         result_tmp_enc_cmt[i] = dict()
#         for j in range(Y.shape[1]):
#             x = int(X[i][j])
#             cmt, ct, idxi, idxj = pool.apply(func=fefo_enc_wrap,
#                                              args=(x, i, j))
#             result_tmp_enc_cmt[idxi][idxj] = cmt
#             result_tmp_enc_ct[idxi][idxj] = ct
#     pool.close()
#     pool.join()
#
#     pool = multiprocessing.Pool()
#     t2 = time.clock()
#     for i in range(Y.shape[0]):
#         result_tmp_key[i] = dict()
#         for j in range(Y.shape[1]):
#             y = int(Y[i][j])
#             res, idxi, idxj = pool.apply(func=fefo_key_wrap,
#                                          args=(result_tmp_enc_cmt[i][j], op, y, i, j))
#             result_tmp_key[idxi][idxj] = res
#     pool.close()
#     pool.join()
#
#     pool = multiprocessing.Pool()
#     t3 = time.clock()
#     for i in range(Y.shape[0]):
#         result_tmp_dec[i] = dict()
#         for j in range(Y.shape[1]):
#             x = int(X[i][j])
#             y = int(Y[i][j])
#             if op == fefo.OPERATION_SYMBOL_ADDITION:
#                 max_prod = abs(x + y) + 1
#             elif op == fefo.OPERATION_SYMBOL_SUBTRACT:
#                 max_prod = abs(x - y) + 1
#             elif op == fefo.OPERATION_SYMBOL_MULTIPLICATION:
#                 max_prod = abs(x * y) + 1
#             elif op == fefo.OPERATION_SYMBOL_DIVISION:
#                 max_prod = int(abs(x / y)) + 1
#             res, idxi, idxj = pool.apply(func=fefo_dec_wrap,
#                                          args=(
#                                              result_tmp_enc_ct[i][j],
#                                              result_tmp_key[i][j], op, y, i, j, max_prod))
#             result_tmp_dec[idxi][idxj] = res
#     pool.close()
#     pool.join()
#     t4 = time.clock()
#
#     enc_cost = (t2 - t1) * 1000
#     key_cost = (t3 - t2) * 1000
#     dec_cost = (t4 - t3) * 1000
#
#     for i in range(Y.shape[0]):
#         for j in range(Y.shape[1]):
#             result[i][j] = result_tmp_dec[i][j]
#     return result, enc_cost, key_cost, dec_cost
#
#
# def smc_fundamental_operation_parallel_float(X, Y, op, precision_x=1000, precision_y=1000):
#     X_tmp = (X * precision_x).astype(int)
#     Y_tmp = (Y * precision_y).astype(int)
#     res = smc_fundamental_operation_parallel(X_tmp, Y_tmp, op)
#     if op == 'addition' or op == 'subtract':
#         return res / precision_x
#     elif op == 'multiplication':
#         return res / (precision_x * precision_y)
#     else:
#         return res


