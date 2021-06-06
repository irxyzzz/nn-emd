import os
import random
import logging
from pytest import fixture

from nn.utils import timer
from crypto.utils import load_dlog_table_config
from crypto.sife_dynamic import SIFEDynamic
from crypto.sife_dynamic import SIFEDynamicTPA
from crypto.sife_dynamic import SIFEDynamicClient
from crypto.utils import generate_config_files

logger = logging.getLogger(__name__)

@fixture
def crypto_config():
    sec_param_config = 'config/sife/sec_param.json'
    dlog_table_config = 'config/sife/dlog_b7.json'
    func_value_bound = 10000000
    sec_param = 256
    if not (os.path.exists(sec_param_config) and os.path.exists(dlog_table_config)):
        logger.debug("could not find the crypto config file, generate a new one")
        generate_config_files(sec_param, sec_param_config,
                              dlog_table_config, func_value_bound)
    return sec_param_config, dlog_table_config

def test_sife_basic():
    logger.info("testing the correctness of basic sife.")
    eta = 5

    max_test_value = 100
    x = [random.randint(0, max_test_value) for i in range(eta)]
    y = [random.randint(0, max_test_value) for i in range(eta)]
    logger.debug("x: %s" % str(x))
    logger.debug("y: %s" % str(y))
    check_prod = sum(map(lambda i: x[i] * y[i], range(eta)))
    logger.debug('original dot product <x,y>: %d' % check_prod)

    sife = SIFEDynamic(eta, sec_param=256)
    sife.setup()
    pk = sife.generate_public_key(len(x))
    ct = sife.encrypt(pk, x)

    sk = sife.generate_private_key(y)
    max_interprod = max_test_value * max_test_value * eta
    with timer('total decryption time:', logger) as t:
        dec_prod = sife.decrypt(pk, sk, y, ct, max_interprod)
        logger.debug('decrypted dot product <x,y>: %d' % dec_prod)
    assert dec_prod == check_prod

def test_sife_basic_with_config(crypto_config):
    logger.info("testing the correctness of sife using config file.")

    eta = 785

    # prepare the test data
    max_test_value = 10
    x = [random.randint(0, max_test_value) for i in range(eta)]
    y = [random.randint(0, max_test_value) for i in range(eta)]
    logger.debug("x: %s" % str(x))
    logger.debug("y: %s" % str(y))
    check_prod = sum(map(lambda i: x[i] * y[i], range(eta)))
    logger.debug('original dot product <x,y>: %d' % check_prod)

    logger.info('loading dlog configuration ...')
    with timer('load dlog config, cost time:', logger) as t:
        dlog = load_dlog_table_config(crypto_config[1])
    logger.info('load dlog configuration DONE')
    sife = SIFEDynamic(eta, sec_param=256,
                       sec_param_config=crypto_config[0], dlog=dlog)
    sife.setup()

    pk = sife.generate_public_key(len(x))
    ct = sife.encrypt(pk, x)
    sk = sife.generate_private_key(y)
    max_interprod = max_test_value * max_test_value * eta
    with timer('total decryption time:', logger) as t:
        dec_prod = sife.decrypt(pk, sk, y, ct, max_interprod)
        logger.debug('decrypted dot product <x,y>: %d' % dec_prod)
    assert dec_prod == check_prod


def test_sife_dynamic():
    logger.info('test dynamic sife in separate roles ...')
    eta = 1000
    sec_param = 256
    max_test_value = 100
    x = [random.randint(0, max_test_value) for i in range(eta)]
    y = [random.randint(0, max_test_value) for i in range(eta)]
    logger.debug("x: %s" % str(x))
    logger.debug("y: %s" % str(y))
    check_prod = sum(map(lambda i: x[i] * y[i], range(eta)))
    logger.debug('original dot product <x,y>: %d' % check_prod)

    logger.info('loading dlog configuration ...')
    with timer('load dlog config, cost time:', logger) as t:
        dlog = load_dlog_table_config(crypto_config[1])
    logger.info('load dlog configuration DONE')
    sife_tpa = SIFEDynamicTPA(eta, sec_param=sec_param, sec_param_config=crypto_config[0])
    sife_tpa.setup()

    sife_enc_client = SIFEDynamicClient(role='enc')
    sife_dec_client = SIFEDynamicClient(role='dec', dlog=dlog)

    pk = sife_tpa.generate_public_key(len(x))
    ct = sife_enc_client.encrypt(pk, x)
    sk = sife_tpa.generate_private_key(y)
    max_interprod = max_test_value * max_test_value * eta
    with timer('total decryption time:', logger) as t:
        dec_prod = sife_dec_client.decrypt(pk, sk, y, ct, max_interprod)
        logger.debug('decrypted dot product <x,y>: %d' % dec_prod)
    assert dec_prod == check_prod
