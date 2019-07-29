
from charm.toolbox.integergroup import IntegerGroupQ
from charm.toolbox.integergroup import integer
# from charm.toolbox.pairinggroup import PairingGroup,ZR,G1
import math
import time
import random

debug = False

class FEMultiplication:

    def __init__(self, groupObj, p=0, q=0):
        global group
        group = groupObj
        if group.groupSetting() == 'integer':
            group.p, group.q, group.r = p, q, 2

    def setup(self, secparam=1024):
        if group.groupSetting() == 'integer':
            if group.p == 0 or group.q == 0:
                group.paramgen(secparam)
            g = group.randomGen()
        msk = group.random()
        mpk = {'g': g, 'h': g ** msk, 'p': group.p}
        return (mpk, msk)

    def keyder(self, msk, commitment, y):
        sk = commitment ** (msk * integer(y, group.p))
        return sk

    def keyder_with_serialize(self, msk, commitment, y):
        sk = group.serialize(group.deserialize(commitment) ** (msk * integer(y, group.p)))
        return sk

    def keyder_int(self, msk, commitment, y):
        sk = integer(commitment, group.p) ** (msk * integer(y, group.p))
        return int(sk)

    def encrypt(self, mpk, x):
        r = group.random()
        commitment = mpk['g'] ** r
        ct = (mpk['h'] ** r) * (mpk['g'] ** x)
        return {'commitment': commitment, 'ct': ct}

    def encrypt_with_serialize(self, mpk, x):
        r = group.random()
        commitment = group.serialize(mpk['g'] ** r)
        ct = group.serialize((mpk['h'] ** r) * (mpk['g'] ** x))
        return {'commitment': commitment, 'ct': ct}

    def encrypt_int(self, mpk, x):
        r = group.random()
        commitment = int(mpk['g'] ** r)
        ct = int((mpk['h'] ** r) * (mpk['g'] ** x))
        return {'commitment': commitment, 'ct': ct}

    def decrypt(self, mpk, ct, sk, y, max_prod):
        # t1 = ct['ct']
        # t2 = t1 ** y
        # t3 = t2 / sk
        g_xy = (ct ** y)/sk
        start = time.clock()
        xy = self.solve_dlog_naive(mpk['g'], g_xy, max_prod)
        if debug:
            print(max_prod)
        # xy = self.bsgs(int(mpk['g']), int(g_xy), int(group.p))
        end = time.clock()
        if debug:
            print("time cost to solve the femul discrete log: " + str(end - start))
        return xy

    def decrypt_with_deserialize(self, mpk, ct, sk, y, max_prod):
        g_xy = (group.deserialize(ct) ** y)/group.deserialize(sk)
        xy = self.solve_dlog_naive(mpk['g'], g_xy, max_prod)
        # xy = self.bsgs(int(mpk['g']), int(g_xy), int(group.p))
        return xy

    def decrypt_int(self, mpk, ct, sk, y, max_prod):
        g_xy = (integer(ct, mpk['p']) ** y)/integer(sk, mpk['p'])
        xy = self.solve_dlog_naive(mpk['g'], g_xy, max_prod)
        return xy

    def solve_dlog_naive(self, g, h, dlog_max):
        """
        Naively attempts to solve for the discrete log x, where g^x = h, via trial and
        error. Assumes that x is at most dlog_max.
        """
        res = None
        for j in range(dlog_max):
            if g ** j == h:
                res = j
        if res == None:
            h = 1 / h
            for i in range(dlog_max):
                if g ** i == h:
                    res = -i
        # if dlog_max > 0:
        #     for j in range(dlog_max):
        #         if g ** j == h:
        #             return j
        # else:
        #     for j in range(-dlog_max):
        #         h = 1 / h
        #         if g ** j == h:
        #             return -j
        return res


class FESubtraction:

    def __init__(self, groupObj, p=0, q=0):
        global group
        group = groupObj
        if group.groupSetting() == 'integer':
            group.p, group.q, group.r = p, q, 2

    def setup(self, secparam=1024):
        if group.groupSetting() == 'integer':
            if group.p == 0 or group.q == 0:
                group.paramgen(secparam)
            g = group.randomGen()
        msk = group.random()
        mpk = {'g': g, 'h': g ** msk, 'p': group.p}
        return (mpk, msk)

    def keyder(self, mpk, msk, commitment, y):
        sk = (commitment ** msk) * (mpk['g'] ** integer(y, group.p))
        return sk

    def encrypt(self, mpk, x):
        r = group.random()
        commitment = mpk['g'] ** r
        ct = (mpk['h'] ** r) * (mpk['g'] ** x)
        return {'commitment': commitment, 'ct': ct}

    def decrypt(self, mpk, ct, sk, max_prod):
        g_sub = ct/sk
        start = time.clock()
        xy = self.solve_dlog_naive(mpk['g'], g_sub, max_prod)
        if debug:
            print(max_prod)
        # xy = self.bsgs(int(mpk['g']), int(g_xy), int(group.p))
        end = time.clock()
        if debug:
            print("time cost to solve the femul discrete log: " + str(end - start))
        return xy

    def solve_dlog_naive(self, g, h, dlog_max):
        """
        Naively attempts to solve for the discrete log x, where g^x = h, via trial and
        error. Assumes that x is at most dlog_max.
        """
        res = None
        for j in range(dlog_max):
            if g ** j == h:
                res = j
        if res == None:
            h = 1 / h
            for i in range(dlog_max):
                if g ** i == h:
                    res = -i
        return res


class FEFundamentalOperation:

    def __init__(self, groupObj, p=0, q=0):
        self.group = groupObj
        if self.group.groupSetting() == 'integer':
            self.group.p = p
            self.group.q = q
            self.group.r = 2
        self.OPERATION_SYMBOL_ADDITION = 'addition'
        self.OPERATION_SYMBOL_SUBTRACT = 'subtract'
        self.OPERATION_SYMBOL_MULTIPLICATION = 'multiplication'
        self.OPERATION_SYMBOL_DIVISION = 'division'

    def setup(self, secparam=1024):
        if self.group.groupSetting() == 'integer':
            if self.group.p == 0 or self.group.q == 0:
                self.group.paramgen(secparam)
            g = self.group.randomGen()
        msk = self.group.random()
        mpk = {'g': g, 'h': g ** msk}
        return mpk, msk

    def keyder(self, mpk, msk, cmt, op, y):
        sk = None
        if op == self.OPERATION_SYMBOL_ADDITION:
            sk = (cmt ** msk) * (mpk['g'] ** (-y))
        elif op == self.OPERATION_SYMBOL_SUBTRACT:
            sk = (cmt ** msk) * (mpk['g'] ** y)
        elif op == self.OPERATION_SYMBOL_MULTIPLICATION:
            sk = (cmt ** (msk * integer(y, self.group.p)))
        elif op == self.OPERATION_SYMBOL_DIVISION:
            # sk = (cmt ** (msk / integer(y, self.group.p)))
            # print((cmt ** (msk / integer(y, self.group.p))))
            sk = (cmt ** msk) ** (integer(y, self.group.p) ** (-1))
            # print(sk)
        return sk

    def keyder_serialize(self, mpk, msk, cmt, op, y):
        cmt = self.group.deserialize(cmt)
        sk = None
        if op == self.OPERATION_SYMBOL_ADDITION:
            sk = (cmt ** msk) * (mpk['g'] ** (-y))
        elif op == self.OPERATION_SYMBOL_SUBTRACT:
            sk = (cmt ** msk) * (mpk['g'] ** y)
        elif op == self.OPERATION_SYMBOL_MULTIPLICATION:
            sk = (cmt ** (msk * integer(y, self.group.p)))
        elif op == self.OPERATION_SYMBOL_DIVISION:
            # sk = (cmt ** (msk / integer(y, self.group.p)))
            # print((cmt ** (msk / integer(y, self.group.p))))
            sk = (cmt ** msk) ** (integer(y, self.group.p) ** (-1))
            # print(sk)
        return self.group.serialize(sk)

    def encrypt(self, mpk, x):
        r = self.group.random()
        cmt = mpk['g'] ** r
        ct = (mpk['h'] ** r) * (mpk['g'] ** x)
        return cmt, ct

    def encrypt_serialize(self, mpk, x):
        r = self.group.random()
        cmt = mpk['g'] ** r
        ct = (mpk['h'] ** r) * (mpk['g'] ** x)
        return self.group.serialize(cmt), self.group.serialize(ct)

    def decrypt(self, mpk, sk, ct, op, y, dlog_max):
        g_f = None
        if op == self.OPERATION_SYMBOL_ADDITION:
            g_f = ct / sk
        elif op == self.OPERATION_SYMBOL_SUBTRACT:
            g_f = ct / sk
        elif op == self.OPERATION_SYMBOL_MULTIPLICATION:
            g_f = (ct ** y) / sk
        elif op == self.OPERATION_SYMBOL_DIVISION:
            # y_tmp = integer(1, self.group.p) / integer(y, self.group.p)
            y_tmp = integer(y, self.group.p) ** (-1)
            g_f = (ct ** y_tmp) / sk
        f = self.solve_dlog_naive(mpk['g'], g_f, dlog_max)
        return f

    def decrypt_serialize(self, mpk, sk, ct, op, y, dlog_max):
        sk = self.group.deserialize(sk)
        ct = self.group.deserialize(ct)
        g_f = None
        if op == self.OPERATION_SYMBOL_ADDITION:
            g_f = ct / sk
        elif op == self.OPERATION_SYMBOL_SUBTRACT:
            g_f = ct / sk
        elif op == self.OPERATION_SYMBOL_MULTIPLICATION:
            g_f = (ct ** y) / sk
        elif op == self.OPERATION_SYMBOL_DIVISION:
            # y_tmp = integer(1, self.group.p) / integer(y, self.group.p)
            y_tmp = integer(y, self.group.p) ** (-1)
            g_f = (ct ** y_tmp) / sk
        f = self.solve_dlog_naive(mpk['g'], g_f, dlog_max)
        return f

    def solve_dlog_naive(self, g, h, dlog_max):
        """
        Naively attempts to solve for the discrete log x, where g^x = h, via trial and
        error. Assumes that x is at most dlog_max.
        """
        res = None
        for j in range(dlog_max):
            if g ** j == h:
                res = j

        if res is None:
            h = 1 / h
            for i in range(dlog_max):
                if g ** i == h:
                    res = -i

        return res