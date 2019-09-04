import random
import time
import math
import os
import logging
import numpy as np
from contextlib import contextmanager

crypto_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.sys.path.insert(0, crypto_path)

from crypto.sife import SIFE

from keras.models import load_model

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

tpa_config_file = '../../config/sife_v785_b8.json'
client_config_file = '../../config/sife_v785_b8_dlog.json'

@contextmanager
def timer(ctx_msg):
    """Helper for measuring runtime"""
    time0 = time.perf_counter()
    yield
    logger.info('[%s][elapsed time: %.2f s]' % (ctx_msg, time.perf_counter() - time0))


def test_generate_config_file():
    logger.info('testing generating config files')
    func_value_bound = 100000000
    # func_value_bound = 100
    secparam = 256
    # 2 parties settings
    sife = SIFE()
    eta = 785
    sife.generate_setup_config(tpa_config_file, secparam, eta)
    sife.generate_dlog_table_config(client_config_file, func_value_bound)


def test_sife_basic():
    logger.info("testing the correctness of basic sife.")

    eta = 5

    # prepare the test data
    max_test_value = 1000
    x = [random.randint(0, max_test_value) for i in range(eta)]
    y = [random.randint(0, max_test_value) for i in range(eta)]
    logger.debug("x: %s" % str(x))
    logger.debug("y: %s" % str(y))
    check_prod = sum(map(lambda i: x[i] * y[i], range(eta)))
    logger.debug('original dot product <x,y>: %d' % check_prod)

    sife = SIFE(eta=eta)
    sife.setup(secparam=256)
    pk = sife.generate_public_key()
    ct = sife.encrypt(pk, x)

    sk = sife.generate_private_key(y)
    max_interprod = max_test_value * max_test_value * eta
    with timer('total decryption time:') as t:
        dec_prod = sife.decrypt(pk, sk, y, ct, max_interprod)
        logger.debug('decrypted dot product <x,y>: %d' % dec_prod)


def test_sife_basic_with_config():
    logger.info("testing the correctness of sife using config file.")

    eta = 16

    # prepare the test data
    max_test_value = 1000
    x = [random.randint(0, max_test_value) for i in range(eta)]
    y = [random.randint(0, max_test_value) for i in range(eta)]
    logger.debug("x: %s" % str(x))
    logger.debug("y: %s" % str(y))
    check_prod = sum(map(lambda i: x[i] * y[i], range(eta)))
    logger.debug('original dot product <x,y>: %d' % check_prod)

    sife = SIFE(tpa_config=tpa_config_file, client_config=client_config_file)

    pk = sife.generate_public_key()
    ct = sife.encrypt(pk, x)

    sk = sife.generate_private_key(y)
    max_interprod = max_test_value * max_test_value * eta
    with timer('total decryption time:') as t:
        dec_prod = sife.decrypt(pk, sk, y, ct, max_interprod)
        logger.debug('decrypted dot product <x,y>: %d' % dec_prod)
