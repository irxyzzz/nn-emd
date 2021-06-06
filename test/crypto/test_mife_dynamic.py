import os
import random
import logging
from pytest import fixture

import numpy as np

from nn.utils import timer
from crypto.mife_dynamic import MIFEDynamic
from crypto.mife_dynamic import MIFEDynamicTPA
from crypto.mife_dynamic import MIFEDynamicClient
from crypto.utils import load_dlog_table_config
from crypto.utils import generate_config_files

logger = logging.getLogger(__name__)


@fixture
def crypto_config():
    sec_param_config = 'config/mife/sec_param.json'
    dlog_table_config = 'config/mife/dlog_b7.json'
    func_value_bound = 10000000
    sec_param = 256
    if not (os.path.exists(sec_param_config) and os.path.exists(dlog_table_config)):
        logger.debug("could not find the crypto config file, generate a new one")
        generate_config_files(sec_param, sec_param_config,
                              dlog_table_config, func_value_bound)
    return sec_param_config, dlog_table_config


@fixture
def parties_input_size():
    parties = {
        'idx-1': 2,
        'idx-2': 3,
        'idx-3': 4
    }
    return parties


def test_mife_basic(parties_input_size):
    logger.info("testing the correctness of basic mife.")

    # prepare the test data
    max_test_value = 100
    x_dict = dict()
    x_vec_count = 0
    x_vec = []
    for idx in parties_input_size.keys():
        x_dict[idx] = [random.randint(0, max_test_value) for _ in range(parties_input_size[idx])]
        x_vec_count = x_vec_count + parties_input_size[idx]
        x_vec = x_vec + x_dict[idx]
    y_vec = [random.randint(0, max_test_value) for _ in range(x_vec_count)]

    logger.debug("x: {}".format(x_vec))
    logger.debug("y: {}".format(y_vec))
    logger.debug('original dot product <x,y>: {}'.format(sum(np.array(x_vec) * np.array(y_vec))))

    mife = MIFEDynamic(sec_param=256, parties=parties_input_size)
    mife.setup()
    ct = dict()
    ct['parties'] = parties_input_size
    ct['ct_dict'] = dict()
    for idx in parties_input_size.keys():
        pk = mife.generate_public_key(idx)
        ct['ct_dict'][idx] = mife.encrypt(pk, x_dict[idx])

    common_pk = mife.generate_common_public_key()
    sk = mife.generate_private_key(y_vec, parties_input_size)
    max_inner_prod = 1000000
    with timer('total decryption time:', logger) as _:
        dec_prod = mife.decrypt(common_pk, sk, y_vec, ct, max_inner_prod)
        logger.debug('decrypted dot product <x,y>: {}'.format(dec_prod))

    assert dec_prod == sum(np.array(x_vec) * np.array(y_vec))


def test_mife_basic_with_config(parties_input_size,
                                crypto_config):
    logger.info("testing the correctness of mife using config file.")

    max_test_value = 100
    x_dict = dict()
    x_vec_count = 0
    x_vec = []
    for idx in parties_input_size.keys():
        x_dict[idx] = [random.randint(0, max_test_value) for _ in range(parties_input_size[idx])]
        x_vec_count = x_vec_count + parties_input_size[idx]
        x_vec = x_vec + x_dict[idx]
    y_vec = [random.randint(0, max_test_value) for i in range(x_vec_count)]

    logger.debug("x: {}".format(x_vec))
    logger.debug("y: {}".format(y_vec))
    logger.debug('original dot product <x,y>: {}'.format(sum(np.array(x_vec) * np.array(y_vec))))

    logger.info('loading dlog configuration ...')
    with timer('load dlog config, cost time:', logger) as t:
        dlog = load_dlog_table_config(crypto_config[1])
    logger.info('load dlog configuration DONE')
    mife = MIFEDynamic(sec_param=256, parties=parties_input_size,
                       sec_param_config=crypto_config[0], dlog=dlog)
    mife.setup()
    ct = dict()
    ct['parties'] = parties_input_size
    ct['ct_dict'] = dict()
    for idx in parties_input_size.keys():
        pk = mife.generate_public_key(idx)
        ct['ct_dict'][idx] = mife.encrypt(pk, x_dict[idx])

    common_pk = mife.generate_common_public_key()
    sk = mife.generate_private_key(y_vec, parties_input_size)
    max_inner_prod = 1000000
    with timer('total decryption time:', logger) as t:
        dec_prod = mife.decrypt(common_pk, sk, y_vec, ct, max_inner_prod)
        logger.debug('decrypted dot product <x,y>: {}'.format(dec_prod))

    assert dec_prod == sum(np.array(x_vec) * np.array(y_vec))


def test_mife_dynamic(parties_input_size,
                      crypto_config):
    logger.info('test dynamic mife ...')
    setup_parties = {
        'idx-1': 2,
        'idx-2': 3,
        'idx-3': 4,
        'idx-4': 4,
        'idx-5': 1,
        'idx-6': 2
    }
    logger.info('loading dlog configuration ...')
    with timer('load dlog config, cost time:', logger) as t:
        dlog = load_dlog_table_config(crypto_config[1])
    logger.info('load dlog configuration DONE')
    mife = MIFEDynamic(sec_param=256, parties=setup_parties,
                       sec_param_config=crypto_config[0], dlog=dlog)
    mife.setup()

    enrolled_parties = {
        'idx-2': 3,
        'idx-3': 4,
        'idx-5': 1
    }

    # prepare the test data
    max_test_value = 100
    x_dict = dict()
    x_vec_count = 0
    x_vec = []
    for idx in enrolled_parties.keys():
        x_dict[idx] = [random.randint(0, max_test_value) for m in range(enrolled_parties[idx])]
        x_vec_count = x_vec_count + enrolled_parties[idx]
        x_vec = x_vec + x_dict[idx]
    y_vec = [random.randint(0, max_test_value) for i in range(x_vec_count)]

    logger.debug("x: %s" % str(x_vec))
    logger.debug("y: %s" % str(y_vec))
    logger.debug('original dot product <x,y>: %d' % int(sum(np.array(x_vec) * np.array(y_vec))))

    ct = dict()
    ct['parties'] = enrolled_parties
    ct['ct_dict'] = dict()
    for idx in enrolled_parties.keys():
        pk = mife.generate_public_key(idx)
        ct['ct_dict'][idx] = mife.encrypt(pk, x_dict[idx])

    common_pk = mife.generate_common_public_key()
    sk = mife.generate_private_key(y_vec, enrolled_parties)
    max_inner_prod = 1000000
    with timer('total decryption time:', logger) as t:
        dec_prod = mife.decrypt(common_pk, sk, y_vec, ct, max_inner_prod)
        logger.debug('decrypted dot product <x,y>: %d' % dec_prod)
    assert dec_prod == sum(np.array(x_vec) * np.array(y_vec))


def test_mife_dynamic_separate(parties_input_size,
                               crypto_config):
    logger.info('test dynamic mife in separate roles ...')
    setup_parties = {
        'idx-1': 2,
        'idx-2': 3,
        'idx-3': 4,
        'idx-4': 4,
        'idx-5': 1,
        'idx-6': 2
    }
    logger.info('loading dlog configuration ...')
    with timer('load dlog config, cost time:', logger) as t:
        dlog = load_dlog_table_config(crypto_config[1])
    logger.info('load dlog configuration DONE')

    mife_tpa = MIFEDynamicTPA(sec_param=256, parties=setup_parties, sec_param_config=crypto_config[0])
    mife_tpa.setup()
    mife_enc_client = MIFEDynamicClient(sec_param=256, role='enc')
    mife_dec_client = MIFEDynamicClient(sec_param=256, role='dec', dlog=dlog)

    enrolled_parties = {
        'idx-2': 3,
        'idx-3': 4,
        'idx-5': 1
    }

    # prepare the test data
    max_test_value = 100
    x_dict = dict()
    x_vec_count = 0
    x_vec = []
    for idx in enrolled_parties.keys():
        x_dict[idx] = [random.randint(0, max_test_value) for m in range(enrolled_parties[idx])]
        x_vec_count = x_vec_count + enrolled_parties[idx]
        x_vec = x_vec + x_dict[idx]
    y_vec = [random.randint(0, max_test_value) for i in range(x_vec_count)]

    logger.debug("x: %s" % str(x_vec))
    logger.debug("y: %s" % str(y_vec))
    logger.debug('original dot product <x,y>: %d' % int(sum(np.array(x_vec) * np.array(y_vec))))

    ct = dict()
    ct['parties'] = enrolled_parties
    ct['ct_dict'] = dict()
    for idx in enrolled_parties.keys():
        pk = mife_tpa.generate_public_key(idx)
        ct['ct_dict'][idx] = mife_enc_client.encrypt(pk, x_dict[idx])

    common_pk = mife_tpa.generate_common_public_key()
    sk = mife_tpa.generate_private_key(y_vec, enrolled_parties)
    max_inner_prod = 1000000
    with timer('total decryption time:', logger) as t:
        dec_prod = mife_dec_client.decrypt(common_pk, sk, y_vec, ct, max_inner_prod)
        logger.debug('decrypted dot product <x,y>: %d' % dec_prod)

    assert dec_prod == sum(np.array(x_vec) * np.array(y_vec))