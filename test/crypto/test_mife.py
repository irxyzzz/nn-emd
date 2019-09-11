import random
import time
import math
import os
import logging
import numpy as np
from contextlib import contextmanager

from crypto.mife import MIFE

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

tpa_config_file = 'config/mife_p2_b8.json'
client_config_file = 'config/mife_p2_b8_dlog.json'

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
    mife = MIFE(2)
    mife.generate_setup_config(tpa_config_file, secparam)
    mife.generate_dlog_table_config(client_config_file, func_value_bound)


def test_mife_basic():
    logger.info("testing the correctness of basic mife.")

    parties = 10
    vec_size = 1

    # prepare the test data
    max_test_value = 1000
    x = []
    for i in range(parties):
        x.append([random.randint(0, max_test_value) for m in range(vec_size)])
    y = [random.randint(0, max_test_value) for i in range(vec_size*parties)]
    logger.debug("x: %s" % str(x))
    logger.debug("y: %s" % str(y))
    check_prod = 0
    for i in range(parties):
        y_i = y[i*vec_size: (i+1)*vec_size]
        check_prod = check_prod + sum(map(lambda j: x[i][j] * y_i[j], range(vec_size)))
    logger.debug('original dot product <x,y>: %d' % check_prod)

    mife = MIFE(parties=parties, vec_size=vec_size)
    mife.setup(secparam=256)
    slot_pk_list = [mife.generate_public_key(i) for i in range(parties)]
    ct_list = [mife.encrypt(slot_pk_list[i], x[i]) for i in range(parties)]

    common_pk = mife.generate_public_key()
    sk = mife.generate_private_key(y)
    max_interprod = max_test_value * max_test_value * vec_size * parties
    with timer('total decryption time:') as t:
        dec_prod = mife.decrypt(common_pk, sk, y, ct_list, max_interprod)
        logger.debug('decrypted dot product <x,y>: %d' % dec_prod)


def test_mife_basic_with_config():
    # 1 passed in 764.24 seconds
    logger.info("testing the correctness of mife using config file.")

    parties = 2
    vec_size = 1

    # prepare the test data
    max_test_value = 1000
    x = []
    for i in range(parties):
        x.append([random.randint(0, max_test_value) for m in range(vec_size)])
    y = [random.randint(0, max_test_value) for i in range(vec_size*parties)]
    logger.debug("x: %s" % str(x))
    logger.debug("y: %s" % str(y))
    check_prod = 0
    for i in range(parties):
        y_i = y[i*vec_size: (i+1)*vec_size]
        check_prod = check_prod + sum(map(lambda j: x[i][j] * y_i[j], range(vec_size)))
    logger.debug('original dot product <x,y>: %d' % check_prod)

    mife = MIFE(tpa_config=tpa_config_file, client_config=client_config_file)
    # mife.setup(secparam=256)
    slot_pk_list = [mife.generate_public_key(i) for i in range(parties)]
    ct_list = [mife.encrypt(slot_pk_list[i], x[i]) for i in range(parties)]

    common_pk = mife.generate_public_key()
    sk = mife.generate_private_key(y)
    max_interprod = max_test_value * max_test_value * vec_size * parties
    with timer('total decryption time:') as t:
        dec_prod = mife.decrypt(common_pk, sk, y, ct_list, max_interprod)
        logger.debug('decrypted dot product <x,y>: %d' % dec_prod)

# if __name__ == '__main__':
    # test_mife()
    # test_compare_solve_dlog()