from crypto.fe_simple import FEInnerProduct
from crypto.fe_simple import FEMultiplication
from crypto.fe_simple import FESubtraction
from charm.toolbox.integergroup import IntegerGroup

import numpy as np
import multiprocessing
import os
import time

debug = False


class SecureComputation():
    def __init__(self, size=1):
        # for test, setting p, q to saving time for paramgen, secparam=256
        self.fe_inner_product = FEInnerProduct(
            IntegerGroup(),
            p=90841625992899044736915068676923590086910503646037290972161689497324782922043,
            q=45420812996449522368457534338461795043455251823018645486080844748662391461021
        )
        self.fe_multiplication = FEMultiplication(
            IntegerGroup(),
            p=90841625992899044736915068676923590086910503646037290972161689497324782922043,
            q=45420812996449522368457534338461795043455251823018645486080844748662391461021
        )
        self.fe_subtraction = FESubtraction(
            IntegerGroup(),
            p=90841625992899044736915068676923590086910503646037290972161689497324782922043,
            q=45420812996449522368457534338461795043455251823018645486080844748662391461021
        )
        self.mpk_feip, self.msk_feip = self.fe_inner_product.setup(size, secparam=256)
        self.mpk_fem, self.msk_fem = self.fe_multiplication.setup(secparam=256)
        self.mpk_fes, self.msk_fes = self.fe_subtraction.setup(secparam=256)
        # cores = multiprocessing.cpu_count()
        # self.pool = multiprocessing.Pool(processes=cores)

    def gen_feip_cipher_x(self, x):
        ct = self.fe_inner_product.encrypt(self.mpk_feip, x)
        return ct

    def gen_fem_cipher_x(self, x):
        ct = self.fe_multiplication.encrypt(self.mpk_fem, x)
        return ct

    def gen_feip_sk_y(self, y):
        sk = self.fe_inner_product.keyder(self.msk_feip, y)
        return sk

    def gen_fem_sk_y(self, commitment, y):
        sk = self.fe_multiplication.keyder(self.msk_fem, commitment, y)
        return sk

    def acquire_inner_product(self, ct, sk, y, max_prod):
        dec_product = self.fe_inner_product.decrypt(self.mpk_feip, ct, sk, y, max_prod)
        return dec_product

    def acquire_multiplication(self, ct, sk, y, max_prod):
        dec_product = self.fe_multiplication.decrypt(self.mpk_fem, ct, sk, y, max_prod)
        return dec_product

    def secure_inner_product(self, X, W):
        '''
        X, W is numpy array
        :param X: n x m, n is row, features; m is column, #samples
        :param W: l x n, l is row, #neuro; n is column, #features parameters
        :return:
        '''
        if debug:
            print("X:")
            print(X)
            print("W:")
            print(W)
        assert X.shape[0] == W.shape[1]
        res = np.zeros((W.shape[0], X.shape[1]))
        for i in range(W.shape[0]):
            for j in range(X.shape[1]):
                x = X[:, j]
                y = W[i, :]
                x_list = x.tolist()
                y_list = y.tolist()
                # if debug:
                #     print("x:" + str(x_list))
                #     print("y:" + str(y_list))
                #     print(type(x_list))
                # if debug:
                #     print("encryption (%s, %s)" % (i,j))
                ct = self.fe_inner_product.encrypt(self.mpk_feip, x_list)
                # if debug:
                #     print("key der (%s, %s)" % (i,j))
                sk = self.fe_inner_product.keyder(self.msk_feip, y_list)
                # if debug:
                #     print("decryption (%s, %s)" % (i,j))
                max_prod = int(np.max(x)) * int(np.max(y)) * x.shape[0]
                dec_product = self.fe_inner_product.decrypt(self.mpk_feip, ct, sk, y_list, max_prod)
                res[i][j] = dec_product
                if debug:
                    print("original inner product: " + str(sum(x*y)))
                    print("  secure inner product: " + str(dec_product))
        return res.astype(int)

    def secure_inner_product_cost(self, X, W):
        '''
        X, W is numpy array
        :param X: n x m, n is row, features; m is column, #samples
        :param W: l x n, l is row, #neuro; n is column, #features parameters
        :return:
        '''
        if debug:
            print("X:")
            print(X)
            print("W:")
            print(W)
        assert X.shape[0] == W.shape[1]
        res = np.zeros((W.shape[0], X.shape[1]))
        enc_cost = 0
        key_cost = 0
        dec_cost = 0
        for i in range(W.shape[0]):
            for j in range(X.shape[1]):
                x = X[:, j]
                y = W[i, :]
                x_list = x.tolist()
                y_list = y.tolist()
                # if debug:
                #     print("x:" + str(x_list))
                #     print("y:" + str(y_list))
                #     print(type(x_list))
                # if debug:
                #     print("encryption (%s, %s)" % (i,j))
                t1 = time.clock()
                ct = self.fe_inner_product.encrypt(self.mpk_feip, x_list)
                t2 = time.clock()
                # if debug:
                #     print("key der (%s, %s)" % (i,j))
                sk = self.fe_inner_product.keyder(self.msk_feip, y_list)
                t3 = time.clock()
                # if debug:
                #     print("decryption (%s, %s)" % (i,j))
                max_prod = int(np.max(x)) * int(np.max(y)) * x.shape[0]
                dec_product = self.fe_inner_product.decrypt(self.mpk_feip, ct, sk, y_list, max_prod)
                t4 = time.clock()
                res[i][j] = dec_product
                if debug:
                    print("original inner product: " + str(sum(x*y)))
                    print("  secure inner product: " + str(dec_product))
                enc_cost += (t2-t1)
                key_cost += (t3-t2)
                dec_cost += (t4-t3)
        return res.astype(int), enc_cost*1000, key_cost*1000, dec_cost*1000

    def secure_inner_product_wrap(self, share_dict, x, y, i, j, max_prod):
        # print("process id %s " % os.getpid())
        # print(x)
        # print(y)
        # print(i)
        # print(j)
        # print(max_prod)
        # x, y, i, j = args[0], args[1], args[2], args[3]
        ct = self.fe_inner_product.encrypt(self.mpk_feip, x)
        sk = self.fe_inner_product.keyder(self.msk_feip, y)
        dec_product = self.fe_inner_product.decrypt(self.mpk_feip, ct, sk, y, max_prod)
        # print(type(dec_product))
        share_dict[i,j] = dec_product
        # q.put([i, j, dec_product])
        # l.append(dec_product)

    def secure_inner_product_multiprocessing(self, X, W):
        '''
        X, W is numpy array
        :param X: n x m, n is row, features; m is column, #samples
        :param W: l x n, l is row, #neuro; n is column, #features parameters
        :return:
        '''
        # if debug:
        #     print("X:")
        #     print(X)
        #     print("W:")
        #     print(W)
        assert X.shape[0] == W.shape[1]
        result = np.zeros((W.shape[0], X.shape[1]))
        manager = multiprocessing.Manager()
        share_dict = manager.dict()
        p_list = []
        for i in range(W.shape[0]):
            for j in range(X.shape[1]):
                x = X[:, j]
                y = W[i, :]
                x_list = x.tolist()
                y_list = y.tolist()
                # if debug:
                #     print("x:" + str(x_list))
                #     print("y:" + str(y_list))
                #     print(type(x_list))
                # if debug:
                #     print(type(x_list[0]))
                #     print(type(i))
                # self.pool.map(self.secure_inner_product_wrap, [x_list, y_list, i, j])
                max_prod = int(np.max(x)) * int(np.max(y)) * x.shape[0]
                proc = multiprocessing.Process(target=self.secure_inner_product_wrap, args=(share_dict, x_list, y_list, i, j, max_prod))
                p_list.append(proc)
                proc.start()
                # tmp = q.get()
                # self.res[tmp[0]][tmp[1]] = tmp[2]
                # if debug:
                #     print("original inner product: " + str(sum(x*y)))
                #     print("  secure inner product: " + str(dec_product))
        for x in p_list:
            x.join()
        for i in range(W.shape[0]):
            for j in range(X.shape[1]):
                result[i][j] = share_dict[i,j]
        return result

    def secure_inner_product_realn_multiprocessing(self, X, W, ceil_X, ceil_W):
        '''
        transfer the random normalize distributed number into int type matrix
        :param X:
        :param W:
        :param ceil_X:
        :param ceil_W:
        :return:
        '''
        X_tmp = (X * ceil_X).astype(int)
        W_tmp = (W * ceil_W).astype(int)
        res = self.secure_inner_product_multiprocessing(X_tmp, W_tmp)
        return res/(ceil_X*ceil_W)

    def secure_inner_product_wrap_enc_cost(self, share_dict_enc, x, i, j):
        ct = self.fe_inner_product.encrypt_int(self.mpk_feip, x)
        share_dict_enc[i,j] = int(ct)

    def secure_inner_product_wrap_key_cost(self, share_dict_key, y, i, j):
        sk, p = self.fe_inner_product.keyder_int(self.msk_feip, y)
        share_dict_key[i,j] = int(sk)

    def secure_inner_product_wrap_dec_cost(self, share_dict_dec, ct, sk, y, i, j, max_prod):
        dec_product = self.fe_inner_product.decrypt_int(self.mpk_feip, ct, sk, y, max_prod)
        share_dict_dec[i,j] = dec_product

    def secure_inner_product_realn_multiprocessing_cost(self, X, W, ceil_X=10, ceil_W=10):
        '''
        transfer the random normalize distributed number into int type matrix
        :param X:
        :param W:
        :param ceil_X:
        :param ceil_W:
        :return:
        '''
        # X = (X * ceil_X).astype(int)
        # W = (W * ceil_W).astype(int)
        assert X.shape[0] == W.shape[1]
        result = np.zeros((W.shape[0], X.shape[1]))
        manager = multiprocessing.Manager()
        share_dict_enc = manager.dict()
        share_dict_key = manager.dict()
        share_dict_dec = manager.dict()
        p_list_enc = []
        p_list_key = []
        p_list_dec = []

        t1 = time.clock()
        for i in range(W.shape[0]):
            for j in range(X.shape[1]):
                x = X[:, j]
                x_list = x.tolist()
                proc = multiprocessing.Process(target=self.secure_inner_product_wrap_enc_cost,
                                               args=(share_dict_enc, x_list, i, j))
                p_list_enc.append(proc)
                proc.start()
        for x in p_list_enc:
            x.join()
        t2 = time.clock()
        for i in range(W.shape[0]):
            for j in range(X.shape[1]):
                y = W[i, :]
                y_list = y.tolist()
                proc = multiprocessing.Process(target=self.secure_inner_product_wrap_key_cost,
                                               args=(share_dict_key, y_list, i, j))
                p_list_key.append(proc)
                proc.start()
        for x in p_list_key:
            x.join()
        t3 = time.clock()
        for i in range(W.shape[0]):
            for j in range(X.shape[1]):
                x = X[:, j]
                y = W[i, :]
                y_list = y.tolist()
                max_prod = int(np.max(x)) * int(np.max(y)) * x.shape[0]
                proc = multiprocessing.Process(target=self.secure_inner_product_wrap_dec_cost,
                                               args=(share_dict_dec, share_dict_enc[i,j], share_dict_key[i,j] ,y_list, i, j, max_prod))
                p_list_dec.append(proc)
                proc.start()
        for x in p_list_dec:
            x.join()
        t4 = time.clock()

        enc_cost = (t2 - t1)*1000
        key_cost = (t3 - t2)*1000
        dec_cost = (t4 - t3)*1000
        for i in range(W.shape[0]):
            for j in range(X.shape[1]):
                result[i][j] = share_dict_dec[i, j]
        return result, enc_cost, key_cost, dec_cost

    def sip_wrap_enc_cost(self, x, i, j):
        ct = self.fe_inner_product.encrypt_int(self.mpk_feip, x)
        self.result_tmp_enc[i, j] = int(ct)

    def sip_wrap_key_cost(self, y, i, j):
        sk, p = self.fe_inner_product.keyder_int(self.msk_feip, y)
        self.result_tmp_key[i, j] = int(sk)

    def sip_wrap_dec_cost(self, ct, sk, y, i, j, max_prod):
        dec_product = self.fe_inner_product.decrypt_int(self.mpk_feip, ct, sk, y, max_prod)
        self.result_tmp_key[(i, j)] = int(dec_product)

    def secure_inner_product_realn_multiprocessing_pool_cost(self, X, W, ceil_X=10, ceil_W=10):
        # X = (X * ceil_X).astype(int)
        # W = (W * ceil_W).astype(int)
        assert X.shape[0] == W.shape[1]
        result = np.zeros((W.shape[0], X.shape[1]))

        # manager = multiprocessing.Manager()

        pool = multiprocessing.Pool()
        self.result_tmp_enc = dict()
        self.result_tmp_key = dict()
        self.result_tmp_dec = dict()

        t1 = time.clock()
        for i in range(W.shape[0]):
            for j in range(X.shape[1]):
                x = X[:, j]
                x_list = x.tolist()
                # print(type(x_list))
                # print(type(i))
                # print(type(j))
                pool.apply(func=self.sip_wrap_enc_cost, args=(x_list, i, j))
        # pool.close()
        # pool.join()
        t2 = time.clock()
        # pool = multiprocessing.Pool()
        for i in range(W.shape[0]):
            for j in range(X.shape[1]):
                y = W[i, :]
                y_list = y.tolist()
                pool.apply(func=self.sip_wrap_key_cost, args=(y_list, i, j))
        pool.close()
        pool.join()
        t3 = time.clock()
        # pool = multiprocessing.Pool()
        for i in range(W.shape[0]):
            for j in range(X.shape[1]):
                x = X[:, j]
                y = W[i, :]
                y_list = y.tolist()
                max_prod = int(np.max(x)) * int(np.max(y)) * x.shape[0]
                pool.apply(func=self.sip_wrap_dec_cost, args=(self.result_tmp_enc[(i,j)],
                                                              self.result_tmp_key[(i,j)],
                                                              y_list, i, j, max_prod))
        pool.close()
        pool.join()
        t4 = time.clock()

        enc_cost = (t2 - t1)*1000
        key_cost = (t3 - t2)*1000
        dec_cost = (t4 - t3)*1000
        for i in range(W.shape[0]):
            for j in range(X.shape[1]):
                result[i][j] = self.result_tmp_dec[(i, j)]
        return result, enc_cost, key_cost, dec_cost

    def secure_inner_product_realn(self, X, W, ceil_X, ceil_W):
        '''
        transfer the random normalize distributed number into int type matrix
        :param X:
        :param W:
        :param ceil_size:
        :return:
        '''
        X_tmp = (X * ceil_X).astype(int)
        W_tmp = (W * ceil_W).astype(int)
        res = self.secure_inner_product(X_tmp, W_tmp)
        return res/(ceil_X*ceil_W)

    def secure_multiplication(self, Y, A, ceil_Y, ceil_A):
        assert Y.shape[0] == A.shape[0]
        assert Y.shape[1] == A.shape[1]
        Y_tmp = (Y * ceil_Y).astype(int)
        A_tmp = (A * ceil_A).astype(int)
        result = np.zeros((Y.shape[0], Y.shape[1]))
        for i in range(Y.shape[0]):
            for j in range(Y.shape[1]):
                # normal multiplication
                # result[i][j] = Y[i][j]*A[i][j]
                y = int(Y_tmp[i][j])
                a = int(A_tmp[i][j])
                ct = self.fe_multiplication.encrypt(self.mpk_fem, y)
                sk = self.fe_multiplication.keyder(self.msk_fem, ct['commitment'], a)
                max_prod = y * a + 1
                dec_mul = self.fe_multiplication.decrypt(self.mpk_fem, ct['ct'], sk, a, max_prod)
                result[i][j] = dec_mul
        return result / (ceil_A*ceil_Y)

    def secure_multiplication_cost(self, Y, A, ceil_Y, ceil_A):
        assert Y.shape[0] == A.shape[0]
        assert Y.shape[1] == A.shape[1]
        Y_tmp = (Y * ceil_Y).astype(int)
        A_tmp = (A * ceil_A).astype(int)
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
                ct = self.fe_multiplication.encrypt(self.mpk_fem, y)
                t2 = time.clock()
                sk = self.fe_multiplication.keyder(self.msk_fem, ct['commitment'], a)
                t3 = time.clock()
                max_prod = y * a + 1
                dec_mul = self.fe_multiplication.decrypt(self.mpk_fem, ct['ct'], sk, a, max_prod)
                t4 = time.clock()
                enc_cost += (t2-t1)*1000
                key_cost += (t3-t2)*1000
                dec_cost += (t4-t3)*1000
                result[i][j] = dec_mul
        return result / (ceil_A*ceil_Y), enc_cost, key_cost, dec_cost

    def secure_multiplication_wrap(self, share_dict, y, a, i, j, max_prod):
        # print("process id %s " % os.getpid())
        ct = self.fe_multiplication.encrypt(self.mpk_fem, y)
        sk = self.fe_multiplication.keyder(self.msk_fem, ct['commitment'], a)
        max_prod = y * a + 1
        dec_mul = self.fe_multiplication.decrypt(self.mpk_fem, ct['ct'], sk, a, max_prod)
        share_dict[i, j] = dec_mul

    def secure_multiplication_multiprocessing(self, Y, A, ceil_Y, ceil_A):
        assert Y.shape[0] == A.shape[0]
        assert Y.shape[1] == A.shape[1]
        Y_tmp = (Y * ceil_Y).astype(int)
        A_tmp = (A * ceil_A).astype(int)
        result = np.zeros((Y.shape[0], Y.shape[1]))
        manager = multiprocessing.Manager()
        share_dict = manager.dict()
        p_list = []
        for i in range(Y.shape[0]):
            for j in range(Y.shape[1]):
                y = int(Y_tmp[i][j])
                a = int(A_tmp[i][j])
                max_prod = y * a + 1
                proc = multiprocessing.Process(target=self.secure_multiplication_wrap,
                                               args=(share_dict, y, a, i, j, max_prod))
                p_list.append(proc)
                proc.start()
        for x in p_list:
            x.join()
        for i in range(Y.shape[0]):
            for j in range(Y.shape[1]):
                result[i][j] = share_dict[i,j]
        return result / (ceil_A*ceil_Y)

    def secure_multiplication_multiprocessing_cost2(self, Y, A, ceil_Y, ceil_A):
        assert Y.shape[0] == A.shape[0]
        assert Y.shape[1] == A.shape[1]
        Y_tmp = (Y * ceil_Y).astype(int)
        A_tmp = (A * ceil_A).astype(int)
        result = np.zeros((Y.shape[0], Y.shape[1]))
        manager = multiprocessing.Manager()
        share_dict = manager.dict()
        p_list = []
        t1 = time.time()
        for i in range(Y.shape[0]):
            for j in range(Y.shape[1]):
                y = int(Y_tmp[i][j])
                a = int(A_tmp[i][j])
                max_prod = y * a + 1
                proc = multiprocessing.Process(target=self.secure_multiplication_wrap,
                                               args=(share_dict, y, a, i, j, max_prod))
                p_list.append(proc)
                proc.start()
        for x in p_list:
            x.join()
        t2 = time.time()

        cost = t2 - t1
        for i in range(Y.shape[0]):
            for j in range(Y.shape[1]):
                result[i][j] = share_dict[i,j]
        return result / (ceil_A*ceil_Y), cost

    def secure_multiplication_enc_wrap(self, y, i, j):
        # print("process id %s " % os.getpid())
        ct = self.fe_multiplication.encrypt_with_serialize(self.mpk_fem, y)
        return ct, i, j

    def secure_multiplication_key_wrap(self, commitment, a, i, j):
        # print("process id %s " % os.getpid())
        sk = self.fe_multiplication.keyder_with_serialize(self.msk_fem, commitment, a)
        return sk, i, j

    def secure_multiplication_dec_wrap(self, ct, sk, a, i, j, max_prod):
        # print("process id %s " % os.getpid())
        dec_mul = self.fe_multiplication.decrypt_with_deserialize(self.mpk_fem, ct, sk, a, max_prod)
        return dec_mul, i, j

    def secure_multiplication_multiprocessing_cost(self, Y, A, ceil_Y, ceil_A):
        assert Y.shape[0] == A.shape[0]
        assert Y.shape[1] == A.shape[1]
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
                res, idxi, idxj = pool.apply_async(func=self.secure_multiplication_enc_wrap,
                                                   args=(y, i, j))
                result_tmp_enc[idxi][idxj] = res
        pool.close()
        pool.join()

        pool = multiprocessing.Pool()
        t2 = time.time()
        for i in range(Y.shape[0]):
            result_tmp_key[i] = dict()
            for j in range(Y.shape[1]):
                a = int(A_tmp[i][j])
                res, idxi, idxj = pool.apply_async(func=self.secure_multiplication_key_wrap,
                                                   args=(result_tmp_enc[i][j]['commitment'], a, i, j))
                result_tmp_key[idxi][idxj] = res
        pool.close()
        pool.join()

        pool = multiprocessing.Pool()
        t3 = time.time()
        for i in range(Y.shape[0]):
            result_tmp_dec[i] = dict()
            for j in range(Y.shape[1]):
                y = int(Y_tmp[i][j])
                a = int(A_tmp[i][j])
                max_prod = y * a + 1
                res, idxi, idxj = pool.apply_async(func=self.secure_multiplication_dec_wrap,
                                                  args=(result_tmp_enc[i][j]['ct'], result_tmp_key[i][j], a, i, j, max_prod))
                result_tmp_dec[idxi][idxj] = res
        pool.close()
        pool.join()
        t4 = time.time()

        enc_cost = (t2-t1)*1000
        key_cost = (t3-t2)*1000
        dec_cost = (t4-t3)*1000

        for i in range(Y.shape[0]):
            for j in range(Y.shape[1]):
                result[i][j] = result_tmp_dec[i,j]
        return result / (ceil_A*ceil_Y), enc_cost, key_cost, dec_cost

    def secure_subtraction(self, X, Y, ceil):
        '''
        X - Y (x for enc, y for key der)
        :param X:
        :param Y:
        :return:
        '''
        assert X.shape[0] == X.shape[0]
        assert X.shape[1] == X.shape[1]
        X_tmp = (X * ceil).astype(int)
        Y_tmp = (Y * ceil).astype(int)
        result = np.zeros((Y.shape[0], Y.shape[1]))
        for i in range(Y.shape[0]):
            for j in range(Y.shape[1]):
                # normal multiplication
                # result[i][j] = X[i][j] - Y[i][j]
                x = int(X_tmp[i][j])
                y = int(Y_tmp[i][j])
                ct = self.fe_subtraction.encrypt(self.mpk_fes, x)
                sk = self.fe_subtraction.keyder(self.mpk_fes, self.msk_fes, ct['commitment'], y)
                max_prod = abs(x - y) + 1
                dec_sub = self.fe_subtraction.decrypt(self.mpk_fes, ct['ct'], sk, max_prod)
                result[i][j] = dec_sub
        return result / ceil
