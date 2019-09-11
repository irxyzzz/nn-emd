import datetime
import logging

import numpy as np

from nn.utils import timer
from nn.smc import Secure2PCClient
from nn.smc import Secure2PCServer
from crypto.sife_dynamic import SIFEDynamicTPA
from crypto.sife_dynamic import SIFEDynamicClient

t_str = str(datetime.datetime.today())
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
    datefmt='%m-%d %H:%M',
    filename="logs/test_smc-" + '-'.join(t_str.split()[:1] + t_str.split()[1].split(':')[:2]) + '.log',
    filemode='w')
logger = logging.getLogger(__name__)

def test_secure2pc():
    logger.info('initialize the crypto system ...')
    sec_param_config_file = 'config/sec_param.json'  # indicate kernel size 5
    dlog_table_config_file = 'config/dlog_b8.json'
    with timer('load sife config file, cost time', logger) as t:
        eta = 1000
        sec_param = 256
        sife_tpa = SIFEDynamicTPA(eta, sec_param=sec_param, sec_param_config=sec_param_config_file)
        sife_tpa.setup()
        sife_enc_client = SIFEDynamicClient(role='enc')
        sife_dec_client = SIFEDynamicClient(role='dec', dlog_table_config=dlog_table_config_file)
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
