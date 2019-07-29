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
import random
import json
import logging
import zlib
import base64

import gmpy2 as gp
import numpy as np

logger = logging.getLogger(__name__)

class MIFE:

    def __init__(self, parties=None, vec_size=1, tpa_config=None, client_config=None,
                 init_dlog_table=False, func_bound=None):
        if tpa_config is not None and client_config is not None:
            self._setup_from_config(tpa_config)
            self._load_dlog_table_config(client_config)
            self.setup_flag = True
        else:
            self.setup_flag = False
            # the number of encryption parties
            self.parties = parties
            # the length of each vector (same length for each, at present)
            self.vec_size = vec_size
            self.init_dlog_table = init_dlog_table
            self.func_bound = func_bound
            self.dlog_table = None

    def _random(self, maximum, bits=256):
        rand_function = random.SystemRandom()
        r = gp.mpz(rand_function.getrandbits(bits))
        while r > maximum:
            r = gp.mpz(rand_function.getrandbits(bits))
        return r

    def _randomPrime(self, bits):
        rand_function = random.SystemRandom()
        r = gp.mpz(rand_function.getrandbits(bits))
        r = gp.bit_set(r, bits - 1)
        return gp.next_prime(r)

    def _paramGen(self, bits, r=2):
        while True:
            self.q = self._randomPrime(bits-1)
            self.p = self.q * 2 + 1
            if gp.is_prime(self.p) and gp.is_prime(self.q):
                break;
        self.r = r

    def _randomGen(self, bits=256):
        while True:
            h = self._random(self.p, bits)
            g = gp.powmod(h, self.r, self.p)
            if not g == 1:
                break
        return g

    def _setup_normal(self, secparam=256):
        self._paramGen(secparam)
        g = self._randomGen(secparam)

        W = []
        u = []
        g_W = []

        for i in range(self.parties):
            W_i = []
            u_i = []
            g_W_i = []
            for l in range(self.vec_size):
                w = self._random(self.p)
                W_i.append(w)
                g_W_i.append(gp.powmod(g, w, self.p))
                u_i.append(self._random(self.p))
            W.append(W_i)
            u.append(u_i)
            g_W.append(g_W_i)

        self.mpk = {'g': g, 'p': self.p, 'g_W': g_W,
                    'parties': self.parties, 'vec_size': self.vec_size}
        self.msk = {'W': W, 'u': u}

    def _setup_from_config(self, config_file=None):
        store_dict = None
        with open(config_file, 'r') as infile:
            store_dict = json.load(infile)

            self.p = gp.mpz(store_dict['group']['p'])
            self.q = gp.mpz(store_dict['group']['q'])
            self.r = gp.mpz(store_dict['group']['r'])
            self.parties = store_dict['group']['parties']
            self.vec_size = store_dict['group']['vec_size']

            g = gp.mpz(store_dict['mpk']['g'])
            W = store_dict['msk']['W']
            u = store_dict['msk']['u']
            g_W = store_dict['mpk']['g_W']

            for i in range(self.parties):
                W_i = W[i]
                u_i = u[i]
                g_W_i = g_W[i]
                for l in range(self.vec_size):
                    W_i[l] = gp.mpz(W_i[l])
                    g_W_i[l] = gp.mpz(g_W_i[l])
                    u_i[l] = gp.mpz(u_i[l])

            self.mpk = {'g': g, 'p': self.p, 'g_W': g_W,
                        'parties': self.parties, 'vec_size': self.vec_size}
            self.msk = {'W': W, 'u': u}

    def _json_zip(self, store_dict):

        return base64.b64encode(
            zlib.compress(
                json.dumps(store_dict).encode('utf-8')
            )
        ).decode('ascii')

    def _json_unzip(self, content):
        try:
            dec_compress = zlib.decompress(base64.b64decode(content))
        except Exception:
            raise RuntimeError("Could not decode/unzip the contents")

        try:
            return json.loads(dec_compress)
        except Exception:
            raise RuntimeError("Could interpret the unzipped contents")

    def _load_dlog_table_config(self, config_file):
        with open(config_file, 'r') as infile:
            config_content = infile.read()
            # store_dict = json.load(infile)
            store_dict = self._json_unzip(config_content)

            self.dlog_table = store_dict['dlog_table']
            self.func_bound = store_dict['func_bound']

            assert gp.mpz(store_dict['g']) == gp.mpz(self.mpk['g']), \
                'g in pk does not match dlog_table'

    def _init_dlog_table(self, func_bound):
        g = self.mpk['g']

        self.dlog_table = dict()
        self.func_bound = func_bound
        bound = self.func_bound + 1
        for i in range(bound):
            self.dlog_table[gp.digits(gp.powmod(g, i, self.p))] = i
        for i in range(-1, -bound, -1):
            self.dlog_table[gp.digits(gp.powmod(g, i, self.p))] = i

    def generate_setup_config(self, config_file, secparam):
        self._paramGen(secparam)
        g = self._randomGen(secparam)

        W = []
        u = []
        g_W = []

        for i in range(self.parties):
            W_i = []
            u_i = []
            g_W_i = []
            for l in range(self.vec_size):
                w = self._random(self.p)
                W_i.append(gp.digits(w))
                g_W_i.append(gp.digits(gp.powmod(g, w, self.p)))
                u_i.append(gp.digits(self._random(self.p)))
            W.append(W_i)
            u.append(u_i)
            g_W.append(g_W_i)

        self.mpk = {
            'g': gp.digits(g),
            'p': gp.digits(self.p),
            'g_W': g_W,
            'parties': self.parties,
            'vec_size': self.vec_size
            }
        self.msk = {'W': W, 'u': u}
        group_info = {
            'p': gp.digits(self.p),
            'q': gp.digits(self.q),
            'r': gp.digits(self.r),
            'parties': self.parties,
            'vec_size': self.vec_size
            }

        store_dict = {'mpk': self.mpk, 'msk': self.msk, 'group': group_info}

        with open(config_file, 'w') as outfile:
            json.dump(store_dict, outfile)
            # outfile.write(self._json_zip(store_dict))

        logger.info('Generate setup-config file successfully, see file %s' % config_file)

    def generate_dlog_table_config(self, config_file, func_bound):
        g = gp.mpz(self.mpk['g'])

        self.dlog_table = dict()
        self.func_bound = func_bound
        bound = self.func_bound + 1
        for i in range(bound):
            self.dlog_table[gp.digits(gp.powmod(g, i, self.p))] = i
        for i in range(-1, -bound, -1):
            self.dlog_table[gp.digits(gp.powmod(g, i, self.p))] = i

        store_dict = {
            'g':gp.digits(g),
            'dlog_table': self.dlog_table,
            'func_bound': self.func_bound
        }

        with open(config_file, 'w') as outfile:
            # json.dump(storeage_dict, outfile)
            outfile.write(self._json_zip(store_dict))
        logger.info('Generate dlog_table file successfully, see file %s' % config_file)

    def setup(self, secparam=256):
        if not self.setup_flag:
            self._setup_normal(secparam)
            if self.func_bound is not None:
                self._init_dlog_table(self.func_bound)

    def _generate_common_public_key(self):
        g_W = self.mpk['g_W']
        g_W_digits = [None for i in range(len(g_W))]
        for i in range(len(g_W)):
            g_W_i_digits = [None for j in range(len(g_W[i]))]
            for j in range(len(g_W[i])):
                g_W_i_digits[j] = gp.digits(g_W[i][j])
            g_W_digits[i] = g_W_i_digits
        
        return {
            'g': gp.digits(self.mpk['g']), 
            'p': gp.digits(self.mpk['p']),  
            'g_W': g_W_digits,
            'parties': self.parties, 
            'vec_size': self.vec_size
        }

    def generate_public_key(self, slot_index=None):
        if slot_index is None:
            return self._generate_common_public_key()
        else:
            slot_mpk = self._generate_common_public_key()
            slot_mpk_W_digits = [None for i in range(self.vec_size)]
            slot_mpk_u_digits = [None for i in range(self.vec_size)]
            for i in range(self.vec_size):
                slot_mpk_W_digits[i] = gp.digits(self.msk['W'][slot_index][i])
                slot_mpk_u_digits[i] = gp.digits(self.msk['u'][slot_index][i])
            slot_mpk['W'] = slot_mpk_W_digits
            slot_mpk['u'] = slot_mpk_u_digits
            return slot_mpk

    def generate_private_key(self, y):
        assert len(y) == self.parties * self.vec_size
        
        d = []
        z = gp.mpz(0)
        for i in range(self.parties):
            W_i = self.msk['W'][i]
            y_i = y[i*self.vec_size:(i+1)*self.vec_size]
            yW_i = gp.mpz(0)
            for j in range(self.vec_size):
                yW_i = yW_i + gp.mul(gp.mpz(y_i[j]), W_i[j])
            d.append(gp.digits(yW_i))
            u_i = self.msk['u'][i]
            for j in range(self.vec_size):
                z = z + gp.mul(gp.mpz(y_i[j]), u_i[j])

        return {'d':d, 'z':gp.digits(z)}

    def encrypt(self, slot_pk, x):
        assert len(x) == self.vec_size

        p = gp.mpz(slot_pk['p'])
        g = gp.mpz(slot_pk['g'])
        u = slot_pk['u']
        W = slot_pk['W']

        r = self._random(p)
        t = gp.powmod(g, r, p)
        c = []
        for j in range(len(x)):
            c.append(gp.digits(gp.powmod(g, gp.mpz(x[j]) + gp.mpz(u[j]) + gp.mul(gp.mpz(W[j]), r), p)))

        return {'t': gp.digits(t), 'c': c}

    def decrypt(self, pk, sk, y, ct_list, max_interprod):

        parties = pk['parties']
        vec_size = pk['vec_size']
        p = gp.mpz(pk['p'])
        g = gp.mpz(pk['g'])
        z = gp.mpz(sk['z'])

        g_f = gp.mpz(1)
        for i in range(parties):
            y_i = y[i*vec_size : (i+1)*vec_size]
            c_i = ct_list[i]['c']
            d_i = sk['d'][i]
            t_i = ct_list[i]['t']
            one_party = gp.mpz(1)
            for j in range(vec_size):
                one_party = gp.mul(one_party,
                    gp.powmod(gp.mpz(c_i[j]), gp.mpz(y_i[j]), p))
            g_f = gp.mul(g_f, gp.divm(one_party, gp.powmod(gp.mpz(t_i), gp.mpz(d_i), p), p))
        g_f = gp.divm(g_f, gp.powmod(g, z, p), p)

        if g_f == gp.mpz(0):
            print(g_f)

        f = self._solve_dlog(p, g, g_f, max_interprod)
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

    def encrypt_weights(self, pk, x_weights, precision=None):
        p = gp.mpz(pk['p'])
        g = gp.mpz(pk['g'])
        u = gp.mpz(pk['u'][0]) # as vec_size is set as 1
        W = gp.mpz(pk['W'][0]) # as vec_size is set as 1

        r = self._random(p)
        t = gp.powmod(g, r, p)

        if precision is not None:
            for i in range(len(x_weights)):
                x_weights[i] = (x_weights[i] * pow(10, precision)).astype(int)

        cipher_weights= list()
        length_weights = len(x_weights)
        for w in range(length_weights):
            weight_vec = x_weights[w]
            new_weight_vec = np.array(x_weights[w], dtype='object')
            for i, weight in np.ndenumerate(weight_vec):
                c = gp.powmod(g, gp.mpz(int(weight)) + u + gp.mul(W, r), p)
                new_weight_vec[i] = c
            cipher_weights.append(new_weight_vec)

        return {'t': t, 'c': cipher_weights}

    def _decrypt_preprocess(self, ct_list):
        parties = len(ct_list)
        adjust_ct_list = dict()
        adjust_ct_list['t'] = [None for i in range(parties)]
        for k in range(parties):
            adjust_ct_list['t'][k] = ct_list[k]['t']

        ct_weights = ct_list[0]['c']
        collect_ct_weights = list()
        for i in range(len(ct_weights)):
            collect_ct_weights_vec = np.array(ct_weights[i], dtype='object')
            for j, weight in np.ndenumerate(ct_weights[i]):
                collect_ct_weights_vec[j] = [ct_list[k]['c'][i][j] for k in range(parties)]
            collect_ct_weights.append(collect_ct_weights_vec)
        adjust_ct_list['c'] = collect_ct_weights

        return adjust_ct_list

    def decrypt_weights(self, pk, sk, y, ct_list, max_interprod, precision=None):
        parties = pk['parties']
        vec_size = pk['vec_size']
        p = gp.mpz(pk['p'])
        g = gp.mpz(pk['g'])
        z = gp.mpz(sk['z'])

        ct_list = self._decrypt_preprocess(ct_list)

        cipher_weights = ct_list['c']
        decipher_weights = list()
        for i in range(len(cipher_weights)):
            cipher_weight_vec = cipher_weights[i]
            decipher_weight_vec = np.array(cipher_weights[i], dtype='object')
            for j, weight in np.ndenumerate(cipher_weight_vec):
                g_f = gp.mpz(1)
                for k in range(parties):
                    d = gp.mpz(sk['d'][k])
                    t = gp.mpz(ct_list['t'][k])
                    g_f = gp.mul(g_f, 
                        gp.divm(
                            gp.powmod(gp.mpz(weight[k]), gp.mpz(y[k]), p), 
                            gp.powmod(t, d, p),
                            p
                        ))
                    # g_f = g_f * (weight[k] ** integer(y[k], self.group.p)) / (t ** d)
                # g_f = g_f / (mpk['g'] ** sk['z'])
                g_f = gp.divm(g_f, gp.powmod(g, z, p), p)
                decipher_weight_vec[j] = self._solve_dlog(p, g, g_f, max_interprod)
            decipher_weights.append(np.array(decipher_weight_vec,dtype=float)/float(self.parties))

        if precision is not None:
            for i in range(len(decipher_weights)):
                decipher_weights[i] = (decipher_weights[i] / pow(10, precision)).astype(float)

        return decipher_weights





