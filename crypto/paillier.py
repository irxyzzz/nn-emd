'''
General and Threshold Paillier (Public-Key)
| From "A Generalisation, a Simplification and some Applications of Paillier’s 
|   Probabilistic Public-Key System"
| Published in: PKC 2001
| By: Ivan Damgård and Mats Jurik
| URL: https://link.springer.com/chapter/10.1007/3-540-44586-2_9

* type:     public-key encryption
* setting:  Integer based

:Authors:   Runhua Xu
:Date:      5/2019
'''
import math
import sys
import random

import gmpy2 as gp
import numpy as np


debug = True

class GeneralPaillier():

    def __init__(self):
        self.NEGTIVE_THRESHOLD = gp.mpz(sys.maxsize)

    def _randomPrime(self, bits):
        rand_function = random.SystemRandom()
        r = gp.mpz(rand_function.getrandbits(bits))
        r = gp.bit_set(r, bits - 1)
        return int(gp.next_prime(r))

    def _paramGen(self, bits):
        p = q = n = None
        while True:
            p, q = self._randomPrime(bits), self._randomPrime(bits)
            if gp.is_prime(p) and gp.is_prime(q) and gp.gcd(p*q, (p-1)*(q-1)) == 1:
                break
        n = p * q

        return (p,q,n)

    def _random(self, maximum, bits=256):
        rand_function = random.SystemRandom()
        r = gp.mpz(rand_function.getrandbits(bits))
        return r

    def _lcm(self, p1, p2):
        return gp.lcm(p1, p2)

    def keygen(self, secparam=256, s=1):
        (p, q, n) = self._paramGen(secparam)
        n_pow_s1 = n ** (s + 1)
        n_pow_s = n ** s
        
        g = n + 1
        lamda = self._lcm(p - 1, q - 1)
        self.mpk = {'n': n, 'n_pow_s1': n_pow_s1, 'n_pow_s': n_pow_s, 'g': g}
        self.msk = {'d': lamda}

        return self.mpk, self.msk

    def encrypt(self, pk, i):
        g = gp.mpz(pk['g'])
        n_pow_s = gp.mpz(pk['n_pow_s'])
        n_pow_s1 = gp.mpz(pk['n_pow_s1'])

        r = self._random(n_pow_s1)
        assert r != gp.mpz(0), 'cannot use 0 as random number'

        # c =(g ** i) * (r ** n_pow_s)
        g_pow_i = gp.powmod(g, i, n_pow_s1)
        r_pow_ns = gp.powmod(r, n_pow_s, n_pow_s1)
        c = gp.t_mod(gp.mul(g_pow_i, r_pow_ns), n_pow_s1)

        return gp.digits(c)

    def decrypt(self, pk, sk, ct):
        n_pow_s1 = gp.mpz(pk['n_pow_s1'])
        n_pow_s = gp.mpz(pk['n_pow_s'])
        c = gp.mpz(ct)
        d = gp.mpz(sk['d'])

        c_pow_d = gp.powmod(c, d, n_pow_s1) - 1
        cd_div_ns = gp.t_div(c_pow_d, n_pow_s)
        d_invert = gp.invert(d, n_pow_s)
        dec = gp.t_mod(gp.mul(d_invert, cd_div_ns), n_pow_s)
        if dec > self.NEGTIVE_THRESHOLD:
            dec = gp.sub(dec, n_pow_s)
            # print(dec)

        # res = -dec if sign else dec
        return int(dec) 
        
    def encrypt_float(self, pk, f, precision=3):
        f_i = int(f*pow(10,precision))
        return self.encrypt(pk, f_i)

    def decrypt_float(self, pk, sk, ct, precision=3):
        dec = self.decrypt(pk, sk, ct)
        return float(dec)/pow(10, precision)

    def decrypt_float_list(self, pk, sk, ct_list, precision=3):
        dec_list = []
        for i in range(len(ct_list)):
            dec_list.append(self.decrypt_float(pk, sk, ct_list[i], precision))
        return dec_list



        
    def _fuze_basic(self, pk, ct_list):
        n_pow_s1 = gp.mpz(pk['n_pow_s1'])

        res = gp.mpz(1)
        for i in range(len(ct_list)):
            res = gp.mul(res, gp.mpz(ct_list[i]))
        res = gp.t_mod(res, n_pow_s1)

        return gp.digits(res)

    def fuze(self, pk, ct_list):
        n_pow_s1 = gp.mpz(pk['n_pow_s1'])

        if not isinstance(ct_list[0], tuple):
            return self._fuze_basic(pk, ct_list)
        else:
            fuzed_ct = gp.mpz(1)
            for i in range(len(ct_list)):
                sign = ct_list[i][0]
                if sign == gp.mpz(1) or sign == gp.mpz(0):
                    fuzed_ct = fuzed_ct * gp.mpz(ct_list[i][1])
                elif sign == gp.mpz(-1):
                    fuzed_ct = fuzed_ct * gp.mpz(ct_list[i][2])
            return gp.digits(fuzed_ct % n_pow_s1)

    # def ext_fuze_two(self, pk, ct_list):
    #     assert isinstance(ct_list[0], tuple), 'ct should in tuple format'
    #     assert len(ct_list) == 2, 'only fuze two elements'
    #
    #     n_pow_s1 = gp.mpz(pk['n_pow_s1'])
    #
    #     if ct_list[0][0] == ct_list[1][0]:
    #         fuzed_ct_sign = gp.mpz(ct_list[0][0])
    #         fuzed_ct_pos = (gp.mpz(ct_list[0][1]) * gp.mpz(ct_list[1][1])) % n_pow_s1
    #         fuzed_ct_neg = (gp.mpz(ct_list[0][2]) * gp.mpz(ct_list[1][2])) % n_pow_s1
    #         return (fuzed_ct_sign, fuzed_ct_pos, fuzed_ct_neg)
    #     else:
    #         fuzed_ct_sign = gp.mpz(ct_list[0][0]) * gp.mpz(ct_list[1][0])
    #         # TODO: issues: exchange, but how to check the pos/neg
    #         if gp.mpz(ct_list[0][0]) == gp.mpz(-1):
    #             fuzed_ct_1 = (gp.mpz(ct_list[0][1]) * gp.mpz(ct_list[1][2])) % n_pow_s1
    #             fuzed_ct_2 = (gp.mpz(ct_list[0][2]) * gp.mpz(ct_list[1][1])) % n_pow_s1
    #         return (fuzed_ct_sign, fuzed_ct_1, fuzed_ct_2)

        # for i in range(1, len(ct_list)):
        #     if fuzed_ct_sign == ct_list[i][0]:
        #         fuzed_ct_sign = fuzed_ct_sign * gp.mpz(ct_list[i][0])
        #         fuzed_ct_pos = fuzed_ct_pos * gp.mpz(ct_list[i][1]) % n_pow_s1
        #         fuzed_ct_neg = fuzed_ct_neg * gp.mpz(ct_list[i][2]) % n_pow_s1
        #     else:
        #         fuzed_ct_sign = fuzed_ct_sign * gp.mpz(ct_list[i][0])
        #         fuzed_ct_pos = fuzed_ct_pos * gp.mpz(ct_list[i][2]) % n_pow_s1
        #         fuzed_ct_neg = fuzed_ct_neg * gp.mpz(ct_list[i][1]) % n_pow_s1
        # return (fuzed_ct_sign, fuzed_ct_pos, fuzed_ct_neg)

    def ext_encrypt(self, pk, i):
        '''
        encrypt i with ct of potistive i and negative i
        '''
        if i == 0:
            return (gp.mpz(1), self.encrypt(pk, 0), self.encrypt(pk, 0))
        return ((i // np.abs(i)) * gp.mpz(1), self.encrypt(pk, abs(i)), self.encrypt(pk, -abs(i)))

    def ext_decrypt(self, pk, sk, ct):        
        # print('ext_decrypt')
        # print(ct)

        if not isinstance(ct, tuple):
            return self.decrypt(pk, sk, ct)

        n_pow_s1 = gp.mpz(pk['n_pow_s1'])
        n_pow_s = gp.mpz(pk['n_pow_s'])
        c = gp.mpz(ct[1])
        d = gp.mpz(sk['d'])

        c_pow_d = gp.powmod(c, d, n_pow_s1) - 1
        cd_div_ns = gp.t_div(c_pow_d, n_pow_s)
        d_invert = gp.invert(d, n_pow_s)
        dec = gp.t_mod(gp.mul(d_invert, cd_div_ns), n_pow_s)
        if dec > self.NEGTIVE_THRESHOLD:
            dec = gp.sub(dec, n_pow_s)
            # print(dec)
        dec = dec * ct[0]
        return int(dec) 

    def ext_encrypt_float(self, pk, f, precision=3):
        f_i = int(f*pow(10,precision))
        return self.ext_encrypt(pk, f_i)

    def ext_decrypt_float(self, pk, sk, ct, precision=3):
        dec = self.ext_decrypt(pk, sk, ct)
        return float(dec)/pow(10, precision)

    def ext_decrypt_float_list(self, pk, sk, ct_list, precision=3):
        dec_list = []
        for i in range(len(ct_list)):
            dec_list.append(self.ext_decrypt_float(pk, sk, ct_list[i], precision))
        return dec_list

    def fuze_pt_ct(self, pk, ext_ct, i):
        # n_pow_s1 = gp.mpz(pk['n_pow_s1'])
        # print('fuze_pt_ct')
        # print(ext_ct)
        # print(i)

        fuzed_ct = None
        if i == 0:
            fuzed_ct = self.ext_encrypt(pk, 0)
            return fuzed_ct

        assert isinstance(ext_ct, tuple), 'cipher should be in tuple format'

        accumlated_ct_pos_list = []
        for k in range(abs(i)):
            accumlated_ct_pos_list.append(ext_ct[1])
        accumlated_ct_pos = self.fuze(pk, accumlated_ct_pos_list)
        accumlated_ct_neg_list = []
        for k in range(abs(i)):
            accumlated_ct_neg_list.append(ext_ct[2])
        accumlated_ct_neg = self.fuze(pk, accumlated_ct_neg_list)

        fuzed_ct = ((i // np.abs(i)) * ext_ct[0], accumlated_ct_pos, accumlated_ct_neg)
        
        return fuzed_ct

    def fuze_pt_ct_float(self, pk, ext_ct, f, precision=3):
        int_f = int(f*pow(10,precision))
        return self.fuze_pt_ct(pk, ext_ct, int_f)

    def fuze_pt_ct_float_list(self, pk, ct_list, f_list, precision=3):
        assert len(ct_list) == len(f_list)
        fuzed_ct_list = []
        for i in range(len(ct_list)):
            fuzed_ct_list.append(self.fuze_pt_ct_float(pk, ct_list[i], f_list[i], precision))
        return fuzed_ct_list

    def fuze_matrix_ct_list(self, pk, mtx, ct_list, precision=3):
        assert mtx.shape[1] == len(ct_list)
        
        res_list = [None for i in range(mtx.shape[0])]
        for i in range(mtx.shape[0]):
            res_list[i] = self.fuze(pk, self.fuze_pt_ct_float_list(pk, ct_list, mtx[i], precision))

        return res_list

    def ext_fuze_two_ct_list(self, pk, sk, ct_a, ct_b):
        assert len(ct_a) == len(ct_b)
        fuzed_ct_list = self.fuze_two_ct_list(pk, ct_a, ct_b)
        # simulation process here: should be a protocol between A/B and C
        fuzed_ct_list_new = [None for i in range(len(fuzed_ct_list))]
        for i in range (len(fuzed_ct_list)):
            # t1 = self.ext_decrypt(pk, sk, fuzed_ct_list[i])
            # t2 = self.ext_encrypt(pk, t1)
            fuzed_ct_list_new[i] = self.ext_encrypt(pk, self.ext_decrypt(pk, sk, fuzed_ct_list[i]))
        return fuzed_ct_list_new


    def fuze_two_ct_list(self, pk, ct_a, ct_b):
        assert len(ct_a) == len(ct_b)

        fuzed_ct_list = [None for i in range(len(ct_a))]
        for i in range(len(ct_a)):
            fuzed_ct_list[i] = self.fuze(pk, [ct_a[i], ct_b[i]])
        return fuzed_ct_list


















