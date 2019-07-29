# (576, 25)
# (25, 12)
import os
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.insert(0,parentdir)
import time
import random

import numpy as np
from charm.toolbox.integergroup import IntegerGroup
from crypto.fe_simple import FEInnerProduct
from crypto.sife import SIFE




def test_fe_inner_production():
    n = 100
    max_value = 600
    x = [random.randint(0, max_value) for i in range(n)]
    y = [random.randint(0, max_value) for i in range(n)]
    # x = [2, 2, 1]
    # y = [4, 1, 1]
    print("x: " + str(x))
    print("y: " + str(y))

    checkprod = sum(map(lambda i: x[i] * y[i], range(n)))
    print('<x, y>: ' + str(checkprod))
    feip = FEInnerProduct(IntegerGroup())
    (mpk, msk) = feip.setup(n, secparam=256)
    ct = feip.encrypt(mpk, x)
    sk = feip.keyder(msk, y)
    start = time.perf_counter()
    dec_prod = feip.decrypt(mpk, ct, sk, y, max_value*max_value*n)
    end = time.perf_counter()
    print('dec <x,y>: ' + str(dec_prod))
    print('cost time: ' + str(end - start))


def test_generate_config():
    secparam = 256
    n = 25
    f_bound = 100000000
    config_file = "../config/sife_n" + str(n) + "_b8.json"
    start = time.perf_counter()
    sife = SIFE()
    sife.generate_config(n, secparam, config_file, f_bound)
    end = time.perf_counter()
    print('cost time: ' + str(end - start))


def test_sife():
    n = 25
    max_value = 600
    x = [random.randint(0, max_value) for i in range(n)]
    y = [random.randint(0, max_value) for i in range(n)]

    checkprod = sum(map(lambda i: x[i] * y[i], range(n)))
    print('<x, y>: ' + str(checkprod))

    config_file = "../config/sife_n" + str(n) + "_b8.json"

    sife = SIFE()
    start = time.perf_counter()
    mpk, msk = sife.setup_from_config(config_file)
    end = time.perf_counter()
    print('setup cost time: ' + str(end - start))    
    ct = sife.encrypt(mpk, x)
    sk = sife.keyder(msk, y)
    start = time.perf_counter()
    dec_prod = sife.decrypt(mpk, ct, sk, y, max_value*max_value*n)
    end = time.perf_counter()
    print('dec <x,y>: ' + str(dec_prod))
    print('cost time: ' + str(end - start))


def test_compare():
    n = 25
    max_value = 600
    x = [random.randint(0, max_value) for i in range(n)]
    y = [random.randint(0, max_value) for i in range(n)]
    # print("x: " + str(x))
    # print("y: " + str(y))
    checkprod = sum(map(lambda i: x[i] * y[i], range(n)))
    print('<x, y>: ' + str(checkprod))

    print('FE version 1')
    feip = FEInnerProduct(IntegerGroup())
    start = time.perf_counter()
    (mpk, msk) = feip.setup(n, secparam=256)
    end = time.perf_counter()
    print('setup cost time: ' + str(end - start))
    ct = feip.encrypt(mpk, x)
    sk = feip.keyder(msk, y)
    start = time.perf_counter()
    dec_prod = feip.decrypt(mpk, ct, sk, y, max_value*max_value*n)
    end = time.perf_counter()
    print('dec <x,y>: ' + str(dec_prod))
    print('cost time: ' + str(end - start))

    print('FE version 2')
    config_file = "../config/sife_n" + str(n) + "_b8.json"
    sife = SIFE()
    start = time.perf_counter()
    mpk, msk = sife.setup_from_config(config_file)
    end = time.perf_counter()
    print('setup cost time: ' + str(end - start))
    ct = sife.encrypt(mpk, x)
    sk = sife.keyder(msk, y)
    start = time.perf_counter()
    dec_prod = sife.decrypt(mpk, ct, sk, y, max_value*max_value*n)
    end = time.perf_counter()
    print('dec <x,y>: ' + str(dec_prod))
    print('cost time: ' + str(end - start))


if __name__ == "__main__":
    # test_fe_inner_production()
    # test_generate_config()
    # test_sife()
    # test_compare()