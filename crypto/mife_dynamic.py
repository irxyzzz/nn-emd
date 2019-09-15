'''
Multi Input Functional Encryption
| From "Multi-Input Functional Encryption for Inner Products:
|          Function-Hiding Realizations and Constructions without Pairings"
| Published in: CRYPTO 2018
| By: Michel Abdalla, Dario Catalano, Dario Fiore, Romain Gay, and Bogdan Ursu5
| URL: https://eprint.iacr.org/2017/972.pdf

* type:     public-key encryption
* setting:  Integer based

:Authors:   Runhua Xu
:Date:      5/2019
'''
import math
import logging

import gmpy2 as gp
import numpy as np

from crypto.utils import _random
from crypto.utils import _random_generator
from crypto.utils import _param_generator
from crypto.utils import load_sec_param_config

logger = logging.getLogger(__name__)

class MIFEDynamic:

    def __init__(self, sec_param=256, parties=None, sec_param_config=None, dlog=None):
        self.parties = parties # dict with vector size

        if sec_param_config is not None and dlog is not None:
            self.p, self.q, self.r, self.g, self.sec_param = load_sec_param_config(sec_param_config)
            self.dlog_table = dlog['dlog_table']
            self.func_bound = dlog['func_bound']
            assert dlog['g'] == self.g, 'g in dlog table does not match g in sec param'
        else:
            self.p, self.q, self.r = _param_generator(sec_param)
            self.g = _random_generator(sec_param, self.p, self.r)
            self.sec_param = sec_param
            self.dlog_table = None
            self.func_bound = None

    def setup(self):
        w, u, g_w = dict(), dict(), dict()

        for idx in self.parties.keys():
            w_idx, u_idx, g_w_idx = list(), list(), list()
            for l in range(self.parties[idx]):
                rnd = _random(self.p, self.sec_param)
                w_idx.append(rnd)
                g_w_idx.append(gp.powmod(self.g, rnd, self.p))
                u_idx.append(_random(self.p, self.sec_param))
            w[idx] = w_idx
            u[idx] = u_idx
            g_w[idx] = g_w_idx

        self.mpk = {
            'g': self.g,
            'p': self.p,
            'sec_param': int(self.sec_param),
            'g_w': g_w,
            'parties': self.parties
        }
        self.msk = {'w': w, 'u': u}

    def generate_common_public_key(self):
        g_w = self.mpk['g_w']
        g_w_digits = dict()
        for idx in g_w.keys():
            g_w_digits[idx] = [gp.digits(j) for j in g_w[idx]]

        return {
            'g': gp.digits(self.mpk['g']), 
            'p': gp.digits(self.mpk['p']),
            'sec_param': self.mpk['sec_param'],
            'g_w': g_w_digits
        }

    def generate_public_key(self, slot_index=None):
        if slot_index is not None:
            slot_mpk = self.generate_common_public_key()
            if slot_index in self.msk['w'] and slot_index in self.msk['u']:
                slot_mpk['w'] = [gp.digits(i) for i in self.msk['w'][slot_index]]
                slot_mpk['u'] = [gp.digits(i) for i in self.msk['u'][slot_index]]
                return slot_mpk
            else:
                logger.error('the slot id is not found.')
                return None

    def _gen_total_parties_vec_size(self, parties):
        count = 0
        for idx in parties.keys():
            count = count + parties[idx]
        return count

    def _split_vector(self, vec, parties):
        assert len(vec) == self._gen_total_parties_vec_size(parties)

        vec_split = dict()
        i = 0
        for idx in parties.keys():
            vec_split[idx] = vec[i: i+parties[idx]]
            i = i + parties[idx]

        return vec_split

    def generate_private_key(self, vec, parties):
        vec_parties = self._split_vector(vec, parties)

        d = dict()
        z = gp.mpz(0)
        for idx in parties.keys():
            if idx in self.msk['w'] and idx in self.msk['u']:
                w_idx = self.msk['w'][idx]
                u_idx = self.msk['u'][idx]
                vec_idx = vec_parties[idx]
                vec_w_idx = gp.mpz(0)
                for j in range(parties[idx]):
                    vec_w_idx = vec_w_idx + gp.mul(gp.mpz(vec_idx[j]), w_idx[j])
                d[idx] = gp.digits(vec_w_idx)
                for j in range(parties[idx]):
                    z = z + gp.mul(gp.mpz(vec_idx[j]), u_idx[j])
            else:
                logger.error('the id %s in parties is not found.' % idx)

        return {'d': d, 'z': gp.digits(z)}

    def encrypt(self, slot_pk, vec):
        assert len(vec) == len(slot_pk['u'])
        assert len(vec) == len(slot_pk['w'])

        p = gp.mpz(slot_pk['p'])
        g = gp.mpz(slot_pk['g'])
        sec_param = slot_pk['sec_param']
        u = slot_pk['u']
        w = slot_pk['w']

        r = _random(p, sec_param)
        t = gp.digits(gp.powmod(g, r, p))

        c = [gp.digits(gp.powmod(g, gp.mpz(vec[i]) + gp.mpz(u[i]) + gp.mul(gp.mpz(w[i]), r), p)) for i in range(len(vec))]

        return {'t': t, 'c': c}

    def decrypt(self, common_pk, sk, vec, ct, max_inner_prod):
        assert len(ct['ct_dict']) == len(sk['d'])

        p = gp.mpz(common_pk['p'])
        g = gp.mpz(common_pk['g'])
        z = gp.mpz(sk['z'])
        d = sk['d']

        vec_parties = self._split_vector(vec, ct['parties'])
        ct_dict = ct['ct_dict']

        g_f = gp.mpz(1)
        for idx in ct_dict.keys():
            vec_idx = vec_parties[idx]
            c_idx = ct_dict[idx]['c']
            t_idx = ct_dict[idx]['t']
            d_idx = d[idx]

            assert len(vec_idx) == len(c_idx)
            init_idx = gp.mpz(1)
            for j in range(len(c_idx)):
                init_idx = gp.mul(init_idx, gp.powmod(gp.mpz(c_idx[j]), gp.mpz(vec_idx[j]), p))

            g_f = gp.mul(g_f, gp.divm(init_idx, gp.powmod(gp.mpz(t_idx), gp.mpz(d_idx), p), p))
        g_f = gp.divm(g_f, gp.powmod(g, z, p), p)

        f = self._solve_dlog(p, g, g_f, max_inner_prod)
        return f

    def _solve_dlog(self, p, g, h, dlog_max):
        """
        Attempts to solve for the discrete log x, where g^x = h mod p via
        hash table.
        """
        if self.dlog_table is not None:
            if gp.digits(h) in self.dlog_table:
                return self.dlog_table[gp.digits(h)]
        else:
            return self._solve_dlog_naive(p, g, h, dlog_max)

    def _solve_dlog_naive(self, p, g, h, dlog_max):
        """
        Attempts to solve for the discrete log x, where g^x = h, via
        trial and error. Assumes that x is at most dlog_max.
        """
        res = None
        for j in range(dlog_max):
            if gp.powmod(g, j, p) == gp.mpz(h):
                res = j
                break
        if res == None:
            h = gp.invert(h, p)
            for i in range(dlog_max):
                if gp.powmod(g, i, p) == gp.mpz(h):
                    res = -i
        return res

    def _solve_dlog_bsgs(self, g, h, p):
        """
        Attempts to solve for the discrete log x, where g^x = h mod p,
        via the Baby-Step Giant-Step algorithm.
        """
        m = math.ceil(math.sqrt(p-1)) # phi(p) is p-1, if p is prime
        # store hashmap of g^{1,2,...,m}(mod p)
        hash_table = {pow(g, i, p): i for i in range(m)}
        # precompute via Fermat's Little Theorem
        c = pow(g, m * (p-2), p)
        # search for an equivalence in the table. Giant Step.
        for j in range(m):
            y = (h * pow(c, j, p)) % p
            if y in hash_table:
                return j * m + hash_table[y]

        return None


class MIFEDynamicTPA(object):
    def __init__(self, sec_param=256, parties=None, sec_param_config=None):
        self.parties = parties # dict with vector size

        if sec_param_config is not None:
            self.p, self.q, self.r, self.g, self.sec_param = load_sec_param_config(sec_param_config)
        else:
            self.p, self.q, self.r = _param_generator(sec_param)
            self.g = _random_generator(sec_param, self.p, self.r)
            self.sec_param = sec_param

    def setup(self):
        w, u, g_w = dict(), dict(), dict()

        for idx in self.parties.keys():
            w_idx, u_idx, g_w_idx = list(), list(), list()
            for l in range(self.parties[idx]):
                rnd = _random(self.p, self.sec_param)
                w_idx.append(rnd)
                g_w_idx.append(gp.powmod(self.g, rnd, self.p))
                u_idx.append(_random(self.p, self.sec_param))
            w[idx] = w_idx
            u[idx] = u_idx
            g_w[idx] = g_w_idx

        self.mpk = {
            'g': self.g,
            'p': self.p,
            'sec_param': int(self.sec_param),
            'g_w': g_w,
            'parties': self.parties
        }
        self.msk = {'w': w, 'u': u}

    def generate_common_public_key(self):
        g_w = self.mpk['g_w']
        g_w_digits = dict()
        for idx in g_w.keys():
            g_w_digits[idx] = [gp.digits(j) for j in g_w[idx]]
        return {
            'g': gp.digits(self.mpk['g']),
            'p': gp.digits(self.mpk['p']),
            'sec_param': self.mpk['sec_param'],
            'g_w': g_w_digits
        }

    def generate_public_key(self, slot_index):
        slot_mpk = self.generate_common_public_key()
        if slot_index in self.msk['w'] and slot_index in self.msk['u']:
            slot_mpk['w'] = [gp.digits(i) for i in self.msk['w'][slot_index]]
            slot_mpk['u'] = [gp.digits(i) for i in self.msk['u'][slot_index]]
            return slot_mpk
        else:
            logger.error('the slot id is not found.')
            return None

    def _gen_total_parties_vec_size(self, parties):
        count = 0
        for idx in parties.keys():
            count = count + parties[idx]
        return count

    def _split_vector(self, vec, parties):
        assert len(vec) == self._gen_total_parties_vec_size(parties)

        vec_split = dict()
        i = 0
        for idx in parties.keys():
            vec_split[idx] = vec[i: i+parties[idx]]
            i = i + parties[idx]

        return vec_split

    def generate_private_key(self, vec, parties):
        vec_parties = self._split_vector(vec, parties)

        d = dict()
        z = gp.mpz(0)
        for idx in parties.keys():
            if idx in self.msk['w'] and idx in self.msk['u']:
                w_idx = self.msk['w'][idx]
                u_idx = self.msk['u'][idx]
                vec_idx = vec_parties[idx]
                vec_w_idx = gp.mpz(0)
                for j in range(parties[idx]):
                    vec_w_idx = vec_w_idx + gp.mul(gp.mpz(vec_idx[j]), w_idx[j])
                d[idx] = gp.digits(vec_w_idx)
                for j in range(parties[idx]):
                    z = z + gp.mul(gp.mpz(vec_idx[j]), u_idx[j])
            else:
                logger.error('the id %s in parties is not found.' % idx)

        return {'d': d, 'z': gp.digits(z)}


class MIFEDynamicClient(object):
    def __init__(self, sec_param=256, role='dec', dlog=None):
        if role == 'dec' or role == 'both':
            if dlog is not None:
                self.dlog_table = dlog['dlog_table']
                self.func_bound = dlog['func_bound']
            else:
                self.sec_param = sec_param
                self.dlog_table = None
                self.func_bound = None
        elif role == 'enc':
            self.sec_param = sec_param

    def encrypt(self, slot_pk, vec):
        assert len(vec) <= len(slot_pk['u'])
        assert len(vec) <= len(slot_pk['w'])

        p = gp.mpz(slot_pk['p'])
        g = gp.mpz(slot_pk['g'])
        sec_param = slot_pk['sec_param']
        u = slot_pk['u']
        w = slot_pk['w']

        r = _random(p, sec_param)
        t = gp.digits(gp.powmod(g, r, p))

        c = [gp.digits(gp.powmod(g, gp.mpz(vec[i]) + gp.mpz(u[i]) + gp.mul(gp.mpz(w[i]), r), p)) for i in range(len(vec))]

        return {'t': t, 'c': c}

    def _gen_total_parties_vec_size(self, parties):
        count = 0
        for idx in parties.keys():
            count = count + parties[idx]
        return count

    def _split_vector(self, vec, parties):
        assert len(vec) == self._gen_total_parties_vec_size(parties)

        vec_split = dict()
        i = 0
        for idx in parties.keys():
            vec_split[idx] = vec[i: i+parties[idx]]
            i = i + parties[idx]

        return vec_split

    def decrypt(self, common_pk, sk, vec, ct, max_inner_prod):
        assert len(ct['ct_dict']) == len(sk['d'])

        p = gp.mpz(common_pk['p'])
        g = gp.mpz(common_pk['g'])
        z = gp.mpz(sk['z'])
        d = sk['d']

        vec_parties = self._split_vector(vec, ct['parties'])
        ct_dict = ct['ct_dict']

        g_f = gp.mpz(1)
        for idx in ct_dict.keys():
            vec_idx = vec_parties[idx]
            c_idx = ct_dict[idx]['c']
            t_idx = ct_dict[idx]['t']
            d_idx = d[idx]

            assert len(vec_idx) == len(c_idx)
            init_idx = gp.mpz(1)
            for j in range(len(c_idx)):
                init_idx = gp.mul(init_idx, gp.powmod(gp.mpz(c_idx[j]), gp.mpz(vec_idx[j]), p))

            g_f = gp.mul(g_f, gp.divm(init_idx, gp.powmod(gp.mpz(t_idx), gp.mpz(d_idx), p), p))
        g_f = gp.divm(g_f, gp.powmod(g, z, p), p)

        f = self._solve_dlog(p, g, g_f, max_inner_prod)
        return f

    def _solve_dlog(self, p, g, h, dlog_max):
        """
        Attempts to solve for the discrete log x, where g^x = h mod p via
        hash table.
        """
        if self.dlog_table is not None:
            if gp.digits(h) in self.dlog_table:
                return self.dlog_table[gp.digits(h)]
        else:
            return self._solve_dlog_naive(p, g, h, dlog_max)

    def _solve_dlog_naive(self, p, g, h, dlog_max):
        """
        Attempts to solve for the discrete log x, where g^x = h, via
        trial and error. Assumes that x is at most dlog_max.
        """
        res = None
        for j in range(dlog_max):
            if gp.powmod(g, j, p) == gp.mpz(h):
                res = j
                break
        if res == None:
            h = gp.invert(h, p)
            for i in range(dlog_max):
                if gp.powmod(g, i, p) == gp.mpz(h):
                    res = -i
        return res

    def _solve_dlog_bsgs(self, g, h, p):
        """
        Attempts to solve for the discrete log x, where g^x = h mod p,
        via the Baby-Step Giant-Step algorithm.
        """
        m = math.ceil(math.sqrt(p-1)) # phi(p) is p-1, if p is prime
        # store hashmap of g^{1,2,...,m}(mod p)
        hash_table = {pow(g, i, p): i for i in range(m)}
        # precompute via Fermat's Little Theorem
        c = pow(g, m * (p-2), p)
        # search for an equivalence in the table. Giant Step.
        for j in range(m):
            y = (h * pow(c, j, p)) % p
            if y in hash_table:
                return j * m + hash_table[y]

        return None