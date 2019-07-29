'''
Simple Single Input Functional Encryption
| From "Simple Functional Encryption Schemes for Inner Products"
| Published in: PKC 2015
| By Michel Abdalla, Florian Bourse, Angelo De Caro, and David Pointcheval
| URL: https://eprint.iacr.org/2015/017.pdf

* type:     public-key encryption
* setting:  Integer based

:Authors:   Runhua Xu
:Date:      7/2019
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

class SIFE:

    def __init__(self, tpa_config=None, client_config=None, eta=None,
                 init_dlog_table=False, func_bound=None):
        if tpa_config is not None and client_config is not None:
            self._setup_from_config(tpa_config)
            self._load_dlog_table_config(client_config)
            self.setup_flag = True
        else:
            self.setup_flag = False
            self.eta = eta
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

        self.msk = []
        for i in range(self.eta):
            self.msk.append(self._random(self.p))

        pk = []
        for i in range(self.eta):
            pk.append(gp.powmod(g, self.msk[i], self.p))
        self.mpk = {'p': self.p, 'g': g, 'pk': pk}

    def _setup_from_config(self, config_file=None):
        with open(config_file, 'r') as infile:
            store_dict = json.load(infile)

            self.p = gp.mpz(store_dict['group']['p'])
            self.q = gp.mpz(store_dict['group']['q'])
            self.r = gp.mpz(store_dict['group']['r'])
            self.eta = store_dict['group']['eta']

            self.msk = [None for i in range(self.eta)]
            g = gp.mpz(store_dict['mpk']['g'])
            pk = [None for i in range(self.eta)]
            for i in range(self.eta):
                pk[i] = gp.mpz(store_dict['mpk']['pk'][i])
                self.msk[i] = gp.mpz(store_dict['msk'][i])

            self.mpk = {'p': self.p, 'g': g, 'pk': pk}

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

    def generate_setup_config(self, config_file, secparam, eta):
        self._paramGen(secparam)
        g = self._randomGen(secparam)

        self.msk = []
        for i in range(eta):
            self.msk.append(gp.digits(self._random(self.p)))

        pk = []
        for i in range(eta):
            pk.append(gp.digits(gp.powmod(g, gp.mpz(self.msk[i]), self.p)))
        self.mpk = {'g': gp.digits(g), 'pk': pk}

        group_info = {
            'p': gp.digits(self.p),
            'q': gp.digits(self.q),
            'r': gp.digits(self.r),
            'eta': eta
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

    def generate_public_key(self):
        pk = dict()
        pk['g'] = gp.digits(self.mpk['g'])
        pk['p'] = gp.digits(self.mpk['p'])
        pk['pk'] = list()
        for i in range(self.eta):
            pk['pk'].append(gp.digits(self.mpk['pk'][i]))
        return pk

    def generate_private_key(self, vec):
        assert len(vec) == self.eta

        sk = gp.mpz(0)
        for i in range(self.eta):
            sk = gp.add(sk, gp.mul(self.msk[i], vec[i]))
        return gp.digits(sk)

    def encrypt(self, pk, vec):
        assert len(vec) == self.eta

        p = gp.mpz(pk['p'])
        g = gp.mpz(pk['g'])

        r = self._random(p)
        ct0 = gp.digits(gp.powmod(g, r, p))
        ct_list = []
        for i in range(len(vec)):
            ct_list.append(gp.digits(
                gp.mul(
                    gp.powmod(gp.mpz(pk['pk'][i]), r, p),
                    gp.powmod(g, gp.mpz(int(vec[i])), p)
                )
            ))
        return {'ct0': ct0, 'ct_list': ct_list}

    def decrypt(self, pk, sk, vec, ct, max_innerprod):
        p = gp.mpz(pk['p'])
        g = gp.mpz(pk['g'])

        res = gp.mpz(1)
        for i in range(len(vec)):
            res = gp.mul(
                res,
                gp.powmod(gp.mpz(ct['ct_list'][i]), gp.mpz(vec[i]), p)
            )
        res = gp.t_mod(res, p)
        g_f = gp.divm(res, gp.powmod(gp.mpz(ct['ct0']), gp.mpz(sk), p), p)

        f = self._solve_dlog(p, g, g_f, max_innerprod)

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
            logger.warning("did not find f in dlog table, may cost more time to compute")
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







