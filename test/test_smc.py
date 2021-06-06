import datetime
import logging

import numpy as np

from nn.utils import timer
from nn.smc import Secure2PCClient
from nn.smc import Secure2PCServer
from nn.smc import EnhancedSecure2PCClient
from nn.smc import EnhancedSecure2PCServer
from crypto.utils import load_dlog_table_config
from crypto.sife_dynamic import SIFEDynamicTPA
from crypto.sife_dynamic import SIFEDynamicClient
from crypto.mife_dynamic import MIFEDynamicTPA
from crypto.mife_dynamic import MIFEDynamicClient

logger = logging.getLogger(__name__)

def test_secure2pc():
    logger.info('initialize the crypto system ...')
    sec_param_config_file = 'config/sec_param.json'  # indicate kernel size 5
    dlog_table_config_file = 'config/dlog_b8.json'
    with timer('load sife config file, cost time', logger) as t:
        eta = 1000
        sec_param = 256
        dlog = load_dlog_table_config(dlog_table_config_file)
        sife_tpa = SIFEDynamicTPA(eta, sec_param=sec_param, sec_param_config=sec_param_config_file)
        sife_tpa.setup()
        sife_enc_client = SIFEDynamicClient(sec_param=256, role='enc')
        sife_dec_client = SIFEDynamicClient(sec_param=256, role='dec', dlog=dlog)
        logger.info('the crypto system initialization done!')

    precision_data = 3
    precision_weight = 3

    secure2pc_client = Secure2PCClient(crypto=(sife_tpa, sife_enc_client), precision=precision_data)
    secure2pc_server = Secure2PCServer(crypto=(sife_tpa, sife_dec_client), precision=(precision_data, precision_weight))

    x = np.array(
        [[
            [0.,          0.,          0.,         0.,          0.],
            [0.,          0.,          0.,         0.,          0.],
            [0.,          0.,         -0.41558442, -0.41558442, -0.41558442],
            [0.,          0.,         -0.41558442, -0.41558442, -0.41558442],
            [0.,          0.,         -0.41558442, -0.41558442, -0.41558442],
        ]])

    y = np.array(
        [[
            [0.25199915, - 0.27933214, - 0.25121472, - 0.01450092, - 0.39264217],
            [-0.13325853,  0.02372098, - 0.1099066,   0.07761139, - 0.14452457],
            [0.18324971,  0.07725119, - 0.05726616,  0.18969544,  0.0127556],
            [-0.08113805,  0.19654118, - 0.37077826, 0.20517105, - 0.35461632],
            [-0.0618344, - 0.07832903, - 0.02575814, - 0.20281196, - 0.31189417]
        ]]
    )

    ct = secure2pc_client.execute(x)
    sk = secure2pc_server.request_key(y)
    dec = secure2pc_server.execute(sk, ct, y)
    logger.info('expected result %f' % np.sum(x * y))
    logger.info('executed result %f' % dec)

def test_enhanced_secure2pc_sife():
    logger.info('test enhanced secure2pc in sife setting')
    logger.info('initialize the crypto system ...')
    sec_param_config_file = 'config/sec_param.json'  # indicate kernel size 5
    dlog_table_config_file = 'config/dlog_b8.json'
    with timer('initialize crypto system, cost time', logger) as t:
        eta = 1000
        sec_param = 256
        setup_parties = {
            'id_1': 2,
            'id_2': 5,
            'id_3': 5,
            'id_4': 3,
            'id_5': 5
        }
        logger.info('loading dlog configuration ...')
        dlog = load_dlog_table_config(dlog_table_config_file)
        logger.info('load dlog configuration DONE')
        sife_tpa = SIFEDynamicTPA(eta, sec_param=sec_param, sec_param_config=sec_param_config_file)
        sife_tpa.setup()
        sife_enc_client = SIFEDynamicClient(sec_param=256, role='enc')
        sife_dec_client = SIFEDynamicClient(sec_param=256, role='dec', dlog=dlog)
        mife_tpa = MIFEDynamicTPA(sec_param=256, parties=setup_parties, sec_param_config=sec_param_config_file)
        mife_tpa.setup()
        mife_enc_client = MIFEDynamicClient(sec_param=256, role='enc')
        mife_dec_client = MIFEDynamicClient(sec_param=256, role='dec', dlog=dlog)
        logger.info('the crypto system initialization done!')

    precision_data = 3
    precision_weight = 3

    es2pc_client = EnhancedSecure2PCClient(
        sife=(sife_tpa, sife_enc_client),
        mife=(mife_tpa, mife_enc_client),
        precision=precision_data)
    es2pc_server = EnhancedSecure2PCServer(
        sife=(sife_tpa, sife_dec_client),
        mife=(mife_tpa, mife_dec_client),
        precision=(precision_data, precision_weight))

    x = np.array(
        [[
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., -0.41558442, -0.41558442, -0.41558442],
            [0., 0., -0.41558442, -0.41558442, -0.41558442],
            [0., 0., -0.41558442, -0.41558442, -0.41558442],
        ]])

    y = np.array(
        [[
            [0.25199915, - 0.27933214, - 0.25121472, - 0.01450092, - 0.39264217],
            [-0.13325853, 0.02372098, - 0.1099066, 0.07761139, - 0.14452457],
            [0.18324971, 0.07725119, - 0.05726616, 0.18969544, 0.0127556],
            [-0.08113805, 0.19654118, - 0.37077826, 0.20517105, - 0.35461632],
            [-0.0618344, - 0.07832903, - 0.02575814, - 0.20281196, - 0.31189417]
        ]]
    )

    ct = es2pc_client.execute(x, {'type': 'sife'})
    sk = es2pc_server.request_key(y, {'type': 'sife'})
    dec = es2pc_server.execute(sk, ct, y, {'type': 'sife'})
    logger.info('expected result %f' % np.sum(x * y))
    logger.info('executed result %f' % dec)

def test_enhanced_secure2pc_mife():
    logger.info('test enhanced secure2pc in mife setting')
    logger.info('initialize the crypto system ...')
    sec_param_config_file = 'config/sec_param.json'  # indicate kernel size 5
    dlog_table_config_file = 'config/dlog_b8.json'
    with timer('initialize crypto system, cost time', logger) as t:
        eta = 1000
        sec_param = 256
        setup_parties = {
            'id_1': 2,
            'id_2': 5,
            'id_3': 5,
            'id_4': 3,
            'id_5': 5
        }
        logger.info('loading dlog configuration ...')
        dlog = load_dlog_table_config(dlog_table_config_file)
        logger.info('load dlog configuration DONE')
        sife_tpa = SIFEDynamicTPA(eta, sec_param=sec_param, sec_param_config=sec_param_config_file)
        sife_tpa.setup()
        sife_enc_client = SIFEDynamicClient(sec_param=256, role='enc')
        sife_dec_client = SIFEDynamicClient(sec_param=256, role='dec', dlog=dlog)
        mife_tpa = MIFEDynamicTPA(sec_param=256, parties=setup_parties, sec_param_config=sec_param_config_file)
        mife_tpa.setup()
        mife_enc_client = MIFEDynamicClient(sec_param=256, role='enc')
        mife_dec_client = MIFEDynamicClient(sec_param=256, role='dec', dlog=dlog)
        logger.info('the crypto system initialization done!')

    precision_data = 3
    precision_weight = 3

    es2pc_client = EnhancedSecure2PCClient(
        sife=(sife_tpa, sife_enc_client),
        mife=(mife_tpa, mife_enc_client),
        precision=precision_data)
    es2pc_server = EnhancedSecure2PCServer(
        sife=(sife_tpa, sife_dec_client),
        mife=(mife_tpa, mife_dec_client),
        precision=(precision_data, precision_weight))

    x = np.array(
        [
            [-0.41558442, -0.41558442],
            [0., -0.41558442],
            [0., 0., -0.41558442],
        ])

    y = np.array(
        [
            [0.25199915, - 0.25121472],
            [0.18324971, 0.07725119],
            [-0.0618344, - 0.07832903, - 0.02575814]
        ])

    expt_res = 0
    for i in range(3):
        expt_res = expt_res + sum(np.array(x[i])*np.array(y[i]))

    enrolled_parties = {
        'id_1': 2,
        'id_3': 2,
        'id_4': 3
    }
    ct = dict()
    ct['parties'] = enrolled_parties
    ct['ct_dict'] = dict()
    ct['ct_dict']['id_1'] = es2pc_client.execute(np.array(x[0]), {'type': 'mife', 'id': 'id_1'})
    ct['ct_dict']['id_3'] = es2pc_client.execute(np.array(x[1]), {'type': 'mife', 'id': 'id_3'})
    ct['ct_dict']['id_4'] = es2pc_client.execute(np.array(x[2]), {'type': 'mife', 'id': 'id_4'})

    y_prime = np.array(y[0] + y[1] + y[2])
    sk = es2pc_server.request_key(y_prime, {'type': 'mife', 'parties': enrolled_parties})
    dec = es2pc_server.execute(sk, ct, y_prime, {'type': 'mife'})
    logger.info('expected result %f' % expt_res)
    logger.info('executed result %f' % dec)

