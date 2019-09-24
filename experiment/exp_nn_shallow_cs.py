import sys
import datetime
import logging

import numpy as np
import matplotlib.pyplot as plt

from nn.shallow.nn_shallow_cs import CryptoNNClient
from nn.shallow.nn_shallow_cs import CryptoNNServer
from nn.utils import load_mnist
from nn.utils import load_mnist_size
from nn.utils import timer
from nn.smc import Secure2PCClient
from nn.smc import Secure2PCServer
from nn.smc import EnhancedSecure2PCClient
from nn.smc import EnhancedSecure2PCServer
from crypto.utils import load_dlog_table_config
from crypto.utils import generate_config_files
from crypto.sife_dynamic import SIFEDynamicTPA
from crypto.sife_dynamic import SIFEDynamicClient
from crypto.mife_dynamic import MIFEDynamicTPA
from crypto.mife_dynamic import MIFEDynamicClient

t_str = str(datetime.datetime.today())
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
    datefmt='%m-%d %H:%M',
    filename="logs/" + __name__ + '-' + '-'.join(t_str.split()[:1] + t_str.split()[1].split(':')[:2]) + '.log',
    filemode='w')
logger = logging.getLogger(__name__)

sec_param_config_file = 'config/sec_param.json'
dlog_table_config_file = 'config/dlog_b8.json'

def test_generate_config_files():
    logger.info('generating config files')
    func_value_bound = 100000000
    sec_param = 256
    generate_config_files(sec_param, sec_param_config_file, dlog_table_config_file, func_value_bound)
    logger.info('generating config files -- DONE')

def test_exp_time_one_batch():
    logger.info('experiment for time cost for one mini-batch')

    logger.info('initialize the crypto system ...')
    # sec_param_config_file = 'config/sec_param.json'  # indicate kernel size 5
    # dlog_table_config_file = 'config/dlog_b8.json'
    with timer('initialize crypto system', logger) as t:
        eta = 1250
        sec_param = 256
        setup_parties = {
            'id_1': 200,
            'id_2': 200,
            'id_3': 200,
            'id_4': 200,
            'id_5': 200
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

    precision_data = 0
    precision_weight = 4

    s2pc_client = Secure2PCClient(sife=(sife_tpa, sife_enc_client), precision=precision_data)
    s2pc_server = Secure2PCServer(sife=(sife_tpa, sife_dec_client), precision=(precision_data, precision_weight))

    es2pc_client = EnhancedSecure2PCClient(
        sife=(sife_tpa, sife_enc_client),
        mife=(mife_tpa, mife_enc_client),
        precision=precision_data)
    es2pc_server = EnhancedSecure2PCServer(
        sife=(sife_tpa, sife_dec_client),
        mife=(mife_tpa, mife_dec_client),
        precision=(precision_data, precision_weight))

    X_train, y_train = load_mnist_size('datasets/mnist', size=600)
    X_test, y_test = load_mnist_size('datasets/mnist', size=100, kind='t10k')

    # shuffle
    X_data, y_data = X_train.copy(), y_train.copy()
    idx = np.random.permutation(X_data.shape[0])
    X_data, y_data = X_data[idx], y_data[idx]

    features_splits = np.array_split(range(X_data.shape[1]), len(setup_parties))
    X_data_lst = [X_data[:, idx] for idx in features_splits]

    total_mini_batches = 10

    nn_client = CryptoNNClient(n_output=10, mini_batches=total_mini_batches, n_features=X_data.shape[1],
                               smc=s2pc_client, random_seed=520)
    nn_server = CryptoNNServer(n_output=10, n_features=X_data.shape[1], hidden_layers=[128, 32],
                               l2=0.1, l1=0.0, epochs=1, eta=0.001, alpha=0.01,
                               decrease_const=0.001, mini_batches=total_mini_batches, smc=s2pc_server)
    logger.info('secure2pc setting: client start to encrypt dataset ...')
    ct_feedforward_lst, ct_backpropagation_lst, y_onehot_lst = nn_client.pre_process(X_data, y_data)
    logger.info('secure2pc setting: client encrypting DONE')
    logger.info('secure2pc setting: server start to train ...')
    with timer('training using secure2pc setting - 10 batches', logger) as t:
        nn_server.fit((ct_feedforward_lst, ct_backpropagation_lst), y_onehot_lst, X_test, y_test)
    logger.info('secure2pc setting: server training DONE')

    nn_server_enhanced = CryptoNNServer(n_output=10, n_features=X_data.shape[1], hidden_layers=[128, 32],
                                        l2=0.1, l1=0.0, epochs=1, eta=0.001, alpha=0.01,
                                        decrease_const=0.001, mini_batches=total_mini_batches, smc=es2pc_server)
    logger.info('enhanced secure2pc setting: client start to encrypt dataset ...')
    ct_ff_lst_dict = dict()
    ct_bp_lst_dict = dict()
    x_idx_count = 0
    final_y_onehot_lst = None
    for id in setup_parties.keys():
        if x_idx_count == (len(setup_parties) - 1):
            n_features = X_data_lst[x_idx_count].shape[1] + 1
            nn_client_enhanced = CryptoNNClient(n_output=10, mini_batches=total_mini_batches, n_features=n_features,
                                                smc=es2pc_client, random_seed=520, id=id)
            nn_server_enhanced.register(nn_client_enhanced)
            ct_feedforward_lst, ct_backpropagation_lst, y_onehot_lst = nn_client_enhanced.pre_process(X_data_lst[x_idx_count],
                                                                                             y_data)
            ct_ff_lst_dict[id] = ct_feedforward_lst
            ct_bp_lst_dict[id] = ct_backpropagation_lst
            final_y_onehot_lst = y_onehot_lst
        else:
            n_features = X_data_lst[x_idx_count].shape[1]
            nn_client_enhanced = CryptoNNClient(n_output=10, mini_batches=total_mini_batches, n_features=n_features,
                                                smc=es2pc_client, random_seed=520, id=id)
            nn_server_enhanced.register(nn_client_enhanced)
            ct_feedforward_lst, ct_backpropagation_lst = nn_client_enhanced.pre_process(X_data_lst[x_idx_count])
            ct_ff_lst_dict[id] = ct_feedforward_lst
            ct_bp_lst_dict[id] = ct_backpropagation_lst
        x_idx_count = x_idx_count + 1
    logger.info('enhanced secure2pc setting: client encrypting DONE')

    logger.info('enhanced secure2pc setting: server start to train ...')
    with timer('training using enhanced secure2pc setting - 10 batches', logger) as t:
        nn_server_enhanced.fit((ct_ff_lst_dict, ct_bp_lst_dict), final_y_onehot_lst, X_test, y_test)
    logger.info('enhanced secure2pc setting: server training DONE')

def test_exp_time_one_batch_diff_ds():
    logger.info('experiment for time cost for one mini-batch for different parties')

    logger.info('initialize the crypto system ...')
    # sec_param_config_file = 'config/sec_param.json'  # indicate kernel size 5
    # dlog_table_config_file = 'config/dlog_b8.json'
    with timer('initialize crypto system', logger) as t:
        eta = 1250
        sec_param = 256
        setup_parties = {
            'id_1': 200,
            'id_2': 200,
            'id_3': 200,
            'id_4': 200,
            'id_5': 200,
            'id_6': 200,
            'id_7': 200,
            'id_8': 200,
            'id_9': 200,
            'id_10': 200,
            'id_11': 200,
            'id_12': 200,
            'id_13': 200,
            'id_14': 200,
            'id_15': 200,
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

    precision_data = 0
    precision_weight = 4

    s2pc_client = Secure2PCClient(sife=(sife_tpa, sife_enc_client), precision=precision_data)
    s2pc_server = Secure2PCServer(sife=(sife_tpa, sife_dec_client), precision=(precision_data, precision_weight))

    es2pc_client = EnhancedSecure2PCClient(
        sife=(sife_tpa, sife_enc_client),
        mife=(mife_tpa, mife_enc_client),
        precision=precision_data)
    es2pc_server = EnhancedSecure2PCServer(
        sife=(sife_tpa, sife_dec_client),
        mife=(mife_tpa, mife_dec_client),
        precision=(precision_data, precision_weight))

    X_train, y_train = load_mnist_size('datasets/mnist', size=600)
    X_test, y_test = load_mnist_size('datasets/mnist', size=100, kind='t10k')

    # shuffle
    X_data, y_data = X_train.copy(), y_train.copy()
    idx = np.random.permutation(X_data.shape[0])
    X_data, y_data = X_data[idx], y_data[idx]

    total_mini_batches = 10

    test_parties_lst = [
        {'id_1': 200, 'id_2': 200, 'id_3': 200, 'id_4': 200,'id_5': 200},
        {'id_1': 200, 'id_2': 200, 'id_3': 200, 'id_4': 200, 'id_5': 200, 'id_6': 200, 'id_7': 200},
        {'id_1': 200, 'id_2': 200, 'id_3': 200, 'id_4': 200, 'id_5': 200, 'id_6': 200, 'id_7': 200, 'id_8': 200,
         'id_9': 200},
        {'id_1': 200, 'id_2': 200, 'id_3': 200, 'id_4': 200, 'id_5': 200, 'id_6': 200, 'id_7': 200, 'id_8': 200,
         'id_9': 200, 'id_10': 200, 'id_11': 200},
        {'id_1': 200, 'id_2': 200, 'id_3': 200, 'id_4': 200, 'id_5': 200, 'id_6': 200, 'id_7': 200, 'id_8': 200,
         'id_9': 200, 'id_10': 200, 'id_11': 200, 'id_12': 200, 'id_13': 200},
        {'id_1': 200, 'id_2': 200, 'id_3': 200, 'id_4': 200, 'id_5': 200, 'id_6': 200, 'id_7': 200, 'id_8': 200,
         'id_9': 200, 'id_10': 200, 'id_11': 200, 'id_12': 200, 'id_13': 200, 'id_14': 200, 'id_15': 200}
    ]
    for test_parties in test_parties_lst:
        nn_client = CryptoNNClient(n_output=10, mini_batches=total_mini_batches, n_features=X_data.shape[1],
                                   smc=s2pc_client, random_seed=520)
        nn_server = CryptoNNServer(n_output=10, n_features=X_data.shape[1], hidden_layers=[128, 32],
                                   l2=0.1, l1=0.0, epochs=1, eta=0.001, alpha=0.01,
                                   decrease_const=0.001, mini_batches=total_mini_batches, smc=s2pc_server)
        logger.info('secure2pc setting: client start to encrypt dataset ...')
        with timer('pre-process using secure2pc setting - 10 batches - parties - ' + str(len(test_parties)), logger) as t:
            ct_feedforward_lst, ct_backpropagation_lst, y_onehot_lst = nn_client.pre_process(X_data, y_data)
        logger.info('secure2pc setting: client encrypting DONE')
        logger.info('secure2pc setting: server start to train ...')
        with timer('training using secure2pc setting - 10 batches - parties - ' + str(len(test_parties)), logger) as t:
            nn_server.fit((ct_feedforward_lst, ct_backpropagation_lst), y_onehot_lst, X_test, y_test)
        logger.info('secure2pc setting: server training DONE')

        nn_server_enhanced = CryptoNNServer(n_output=10, n_features=X_data.shape[1], hidden_layers=[128, 32],
                                            l2=0.1, l1=0.0, epochs=1, eta=0.001, alpha=0.01,
                                            decrease_const=0.001, mini_batches=total_mini_batches, smc=es2pc_server)
        logger.info('enhanced secure2pc setting: client start to encrypt dataset ...')
        ct_ff_lst_dict = dict()
        ct_bp_lst_dict = dict()
        x_idx_count = 0
        features_splits = np.array_split(range(X_data.shape[1]), len(test_parties))
        X_data_lst = [X_data[:, idx] for idx in features_splits]
        final_y_onehot_lst = None
        with timer('pre-process using secure2pc setting - 10 batches - parties - ' + str(len(test_parties)), logger) as t:
            for id in test_parties.keys():
                if x_idx_count == (len(test_parties) - 1):
                    n_features = X_data_lst[x_idx_count].shape[1] + 1
                    nn_client_enhanced = CryptoNNClient(n_output=10, mini_batches=total_mini_batches, n_features=n_features,
                                                        smc=es2pc_client, random_seed=520, id=id)
                    nn_server_enhanced.register(nn_client_enhanced)
                    ct_feedforward_lst, ct_backpropagation_lst, y_onehot_lst = nn_client_enhanced.pre_process(X_data_lst[x_idx_count],
                                                                                                     y_data)
                    ct_ff_lst_dict[id] = ct_feedforward_lst
                    ct_bp_lst_dict[id] = ct_backpropagation_lst
                    final_y_onehot_lst = y_onehot_lst
                else:
                    n_features = X_data_lst[x_idx_count].shape[1]
                    nn_client_enhanced = CryptoNNClient(n_output=10, mini_batches=total_mini_batches, n_features=n_features,
                                                        smc=es2pc_client, random_seed=520, id=id)
                    nn_server_enhanced.register(nn_client_enhanced)
                    ct_feedforward_lst, ct_backpropagation_lst = nn_client_enhanced.pre_process(X_data_lst[x_idx_count])
                    ct_ff_lst_dict[id] = ct_feedforward_lst
                    ct_bp_lst_dict[id] = ct_backpropagation_lst
                x_idx_count = x_idx_count + 1
        logger.info('enhanced secure2pc setting: client encrypting DONE')

        logger.info('enhanced secure2pc setting: server start to train ...')
        with timer('training using enhanced secure2pc setting - 10 batches - parties - ' + str(len(test_parties)), logger) as t:
            nn_server_enhanced.fit((ct_ff_lst_dict, ct_bp_lst_dict), final_y_onehot_lst, X_test, y_test)
        logger.info('enhanced secure2pc setting: server training DONE')

def test_exp_time_one_batch_diff_precision():
    logger.info('experiment for time cost for one mini-batch')

    logger.info('initialize the crypto system ...')
    # sec_param_config_file = 'config/sec_param.json'  # indicate kernel size 5
    # dlog_table_config_file = 'config/dlog_b8.json'
    with timer('initialize crypto system', logger) as t:
        eta = 1250
        sec_param = 256
        setup_parties = {
            'id_1': 200,
            'id_2': 200,
            'id_3': 200,
            'id_4': 200,
            'id_5': 200
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

    precision_data = 0
    precision_weight_lst = [2, 3, 4, 5]
    for precision_weight in precision_weight_lst:
        s2pc_client = Secure2PCClient(sife=(sife_tpa, sife_enc_client), precision=precision_data)
        s2pc_server = Secure2PCServer(sife=(sife_tpa, sife_dec_client), precision=(precision_data, precision_weight))

        es2pc_client = EnhancedSecure2PCClient(
            sife=(sife_tpa, sife_enc_client),
            mife=(mife_tpa, mife_enc_client),
            precision=precision_data)
        es2pc_server = EnhancedSecure2PCServer(
            sife=(sife_tpa, sife_dec_client),
            mife=(mife_tpa, mife_dec_client),
            precision=(precision_data, precision_weight))

        X_train, y_train = load_mnist_size('datasets/mnist', size=600)
        X_test, y_test = load_mnist_size('datasets/mnist', size=100, kind='t10k')

        # shuffle
        X_data, y_data = X_train.copy(), y_train.copy()
        idx = np.random.permutation(X_data.shape[0])
        X_data, y_data = X_data[idx], y_data[idx]

        features_splits = np.array_split(range(X_data.shape[1]), len(setup_parties))
        X_data_lst = [X_data[:, idx] for idx in features_splits]

        total_mini_batches = 10

        nn_client = CryptoNNClient(n_output=10, mini_batches=total_mini_batches, n_features=X_data.shape[1],
                                   smc=s2pc_client, random_seed=520)
        nn_server = CryptoNNServer(n_output=10, n_features=X_data.shape[1], hidden_layers=[128, 32],
                                   l2=0.1, l1=0.0, epochs=1, eta=0.001, alpha=0.01,
                                   decrease_const=0.001, mini_batches=total_mini_batches, smc=s2pc_server)
        logger.info('secure2pc setting: client start to encrypt dataset ...')
        ct_feedforward_lst, ct_backpropagation_lst, y_onehot_lst = nn_client.pre_process(X_data, y_data)
        logger.info('secure2pc setting: client encrypting DONE')
        logger.info('secure2pc setting: server start to train ...')
        with timer('training using secure2pc setting - 10 batches - P' + str(precision_weight), logger) as t:
            nn_server.fit((ct_feedforward_lst, ct_backpropagation_lst), y_onehot_lst, X_test, y_test)
        logger.info('secure2pc setting: server training DONE')

        nn_server_enhanced = CryptoNNServer(n_output=10, n_features=X_data.shape[1], hidden_layers=[128, 32],
                                            l2=0.1, l1=0.0, epochs=1, eta=0.001, alpha=0.01,
                                            decrease_const=0.001, mini_batches=total_mini_batches, smc=es2pc_server)
        logger.info('enhanced secure2pc setting: client start to encrypt dataset ...')
        ct_ff_lst_dict = dict()
        ct_bp_lst_dict = dict()
        x_idx_count = 0
        final_y_onehot_lst = None
        for id in setup_parties.keys():
            if x_idx_count == (len(setup_parties) - 1):
                n_features = X_data_lst[x_idx_count].shape[1] + 1
                nn_client_enhanced = CryptoNNClient(n_output=10, mini_batches=total_mini_batches, n_features=n_features,
                                                    smc=es2pc_client, random_seed=520, id=id)
                nn_server_enhanced.register(nn_client_enhanced)
                ct_feedforward_lst, ct_backpropagation_lst, y_onehot_lst = nn_client_enhanced.pre_process(X_data_lst[x_idx_count],
                                                                                                 y_data)
                ct_ff_lst_dict[id] = ct_feedforward_lst
                ct_bp_lst_dict[id] = ct_backpropagation_lst
                final_y_onehot_lst = y_onehot_lst
            else:
                n_features = X_data_lst[x_idx_count].shape[1]
                nn_client_enhanced = CryptoNNClient(n_output=10, mini_batches=total_mini_batches, n_features=n_features,
                                                    smc=es2pc_client, random_seed=520, id=id)
                nn_server_enhanced.register(nn_client_enhanced)
                ct_feedforward_lst, ct_backpropagation_lst = nn_client_enhanced.pre_process(X_data_lst[x_idx_count])
                ct_ff_lst_dict[id] = ct_feedforward_lst
                ct_bp_lst_dict[id] = ct_backpropagation_lst
            x_idx_count = x_idx_count + 1
        logger.info('enhanced secure2pc setting: client encrypting DONE')

        logger.info('enhanced secure2pc setting: server start to train ...')
        with timer('training using enhanced secure2pc setting - 10 batches - P' + str(precision_weight), logger) as t:
            nn_server_enhanced.fit((ct_ff_lst_dict, ct_bp_lst_dict), final_y_onehot_lst, X_test, y_test)
        logger.info('enhanced secure2pc setting: server training DONE')


def test_exp_diff_networks():
    logger.info('experiment for time cost for one mini-batch for different networks')

    logger.info('initialize the crypto system ...')
    sec_param_config_file = 'config/sec_param.json'  # indicate kernel size 5
    dlog_table_config_file = 'config/dlog_b8.json'
    with timer('initialize crypto system', logger) as t:
        eta = 1250
        sec_param = 256
        setup_parties = {
            'id_1': 200,
            'id_2': 200,
            'id_3': 200,
            'id_4': 200,
            'id_5': 200
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

    precision_data = 0
    precision_weight = 4

    s2pc_client = Secure2PCClient(sife=(sife_tpa, sife_enc_client), precision=precision_data)
    s2pc_server = Secure2PCServer(sife=(sife_tpa, sife_dec_client), precision=(precision_data, precision_weight))

    es2pc_client = EnhancedSecure2PCClient(
        sife=(sife_tpa, sife_enc_client),
        mife=(mife_tpa, mife_enc_client),
        precision=precision_data)
    es2pc_server = EnhancedSecure2PCServer(
        sife=(sife_tpa, sife_dec_client),
        mife=(mife_tpa, mife_dec_client),
        precision=(precision_data, precision_weight))

    X_train, y_train = load_mnist_size('datasets/mnist', size=600)
    X_test, y_test = load_mnist_size('datasets/mnist', size=100, kind='t10k')

    # shuffle
    X_data, y_data = X_train.copy(), y_train.copy()
    idx = np.random.permutation(X_data.shape[0])
    X_data, y_data = X_data[idx], y_data[idx]

    features_splits = np.array_split(range(X_data.shape[1]), len(setup_parties))
    X_data_lst = [X_data[:, idx] for idx in features_splits]

    total_mini_batches = 10
    hidden_layers_sets = [
        [256],
        [256, 128],
        [256, 128, 64],
        [256, 128, 64, 32],
        [256, 128, 64, 32, 16]
    ]
    nn_client = CryptoNNClient(n_output=10, mini_batches=total_mini_batches, n_features=X_data.shape[1],
                               smc=s2pc_client, random_seed=520)
    logger.info('secure2pc setting: client start to encrypt dataset ...')
    ct_feedforward_lst, ct_backpropagation_lst, y_onehot_lst = nn_client.pre_process(X_data, y_data)
    logger.info('secure2pc setting: client encrypting DONE')
    for hidden_layers in hidden_layers_sets:
        nn_server = CryptoNNServer(n_output=10, n_features=X_data.shape[1], hidden_layers=hidden_layers,
                                   l2=0.1, l1=0.0, epochs=1, eta=0.001, alpha=0.01,
                                   decrease_const=0.001, mini_batches=total_mini_batches, smc=s2pc_server)
        logger.info('secure2pc setting: server start to train ...')
        with timer('training using secure2pc setting - 10 batches - ' + str(hidden_layers), logger) as t:
            nn_server.fit((ct_feedforward_lst, ct_backpropagation_lst), y_onehot_lst, X_test, y_test)
        logger.info('secure2pc setting: server training DONE')

    logger.info('enhanced secure2pc setting: client start to encrypt dataset ...')
    ct_ff_lst_dict = dict()
    ct_bp_lst_dict = dict()
    x_idx_count = 0
    final_y_onehot_lst = None
    nn_client_enhanced_dict = dict()
    for id in setup_parties.keys():
        if x_idx_count == (len(setup_parties) - 1):
            n_features = X_data_lst[x_idx_count].shape[1] + 1
            nn_client_enhanced = CryptoNNClient(n_output=10, mini_batches=total_mini_batches, n_features=n_features,
                                                smc=es2pc_client, random_seed=520, id=id)
            nn_client_enhanced_dict[id] = nn_client_enhanced
            ct_feedforward_lst, ct_backpropagation_lst, y_onehot_lst = nn_client_enhanced.pre_process(X_data_lst[x_idx_count],
                                                                                             y_data)
            ct_ff_lst_dict[id] = ct_feedforward_lst
            ct_bp_lst_dict[id] = ct_backpropagation_lst
            final_y_onehot_lst = y_onehot_lst
        else:
            n_features = X_data_lst[x_idx_count].shape[1]
            nn_client_enhanced = CryptoNNClient(n_output=10, mini_batches=total_mini_batches, n_features=n_features,
                                                smc=es2pc_client, random_seed=520, id=id)
            nn_client_enhanced_dict[id] = nn_client_enhanced
            ct_feedforward_lst, ct_backpropagation_lst = nn_client_enhanced.pre_process(X_data_lst[x_idx_count])
            ct_ff_lst_dict[id] = ct_feedforward_lst
            ct_bp_lst_dict[id] = ct_backpropagation_lst
        x_idx_count = x_idx_count + 1
    logger.info('enhanced secure2pc setting: client encrypting DONE')
    for hidden_layers in hidden_layers_sets:
        nn_server_enhanced = CryptoNNServer(n_output=10, n_features=X_data.shape[1], hidden_layers=hidden_layers,
                                            l2=0.1, l1=0.0, epochs=1, eta=0.001, alpha=0.01,
                                            decrease_const=0.001, mini_batches=total_mini_batches, smc=es2pc_server)
        for id in nn_client_enhanced_dict.keys():
            nn_server_enhanced.register(nn_client_enhanced_dict[id])

        logger.info('enhanced secure2pc setting: server start to train ...')
        with timer('training using enhanced secure2pc setting - 10 batches - ' + str(hidden_layers), logger) as t:
            nn_server_enhanced.fit((ct_ff_lst_dict, ct_bp_lst_dict), final_y_onehot_lst, X_test, y_test)
        logger.info('enhanced secure2pc setting: server training DONE')

def test_exp_diff_networks_same_hidden_layer():
    logger.info('experiment for time cost for one mini-batch for different networks with same hidden layer')

    logger.info('initialize the crypto system ...')
    sec_param_config_file = 'config/sec_param.json'  # indicate kernel size 5
    dlog_table_config_file = 'config/dlog_b8.json'
    with timer('initialize crypto system', logger) as t:
        eta = 1250
        sec_param = 256
        setup_parties = {
            'id_1': 200,
            'id_2': 200,
            'id_3': 200,
            'id_4': 200,
            'id_5': 200
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

    precision_data = 0
    precision_weight = 4

    s2pc_client = Secure2PCClient(sife=(sife_tpa, sife_enc_client), precision=precision_data)
    s2pc_server = Secure2PCServer(sife=(sife_tpa, sife_dec_client), precision=(precision_data, precision_weight))

    es2pc_client = EnhancedSecure2PCClient(
        sife=(sife_tpa, sife_enc_client),
        mife=(mife_tpa, mife_enc_client),
        precision=precision_data)
    es2pc_server = EnhancedSecure2PCServer(
        sife=(sife_tpa, sife_dec_client),
        mife=(mife_tpa, mife_dec_client),
        precision=(precision_data, precision_weight))

    X_train, y_train = load_mnist_size('datasets/mnist', size=300)
    X_test, y_test = load_mnist_size('datasets/mnist', size=100, kind='t10k')

    # shuffle
    X_data, y_data = X_train.copy(), y_train.copy()
    idx = np.random.permutation(X_data.shape[0])
    X_data, y_data = X_data[idx], y_data[idx]

    features_splits = np.array_split(range(X_data.shape[1]), len(setup_parties))
    X_data_lst = [X_data[:, idx] for idx in features_splits]

    total_mini_batches = 5
    hidden_layers_sets = [[64 for j in range(1, i)] for i in range(1, 32)]
    hidden_layers_sets = hidden_layers_sets[1:]
    nn_client = CryptoNNClient(n_output=10, mini_batches=total_mini_batches, n_features=X_data.shape[1],
                               smc=s2pc_client, random_seed=520)
    logger.info('secure2pc setting: client start to encrypt dataset ...')
    ct_feedforward_lst, ct_backpropagation_lst, y_onehot_lst = nn_client.pre_process(X_data, y_data)
    logger.info('secure2pc setting: client encrypting DONE')
    for hidden_layers in hidden_layers_sets:
        nn_server = CryptoNNServer(n_output=10, n_features=X_data.shape[1], hidden_layers=hidden_layers,
                                   l2=0.1, l1=0.0, epochs=1, eta=0.001, alpha=0.01,
                                   decrease_const=0.001, mini_batches=total_mini_batches, smc=s2pc_server)
        logger.info('secure2pc setting: server start to train ...')
        with timer('training using secure2pc setting - 5 batches - ' + str(len(hidden_layers)) + ' layers', logger) as t:
            nn_server.fit((ct_feedforward_lst, ct_backpropagation_lst), y_onehot_lst, X_test, y_test)
        logger.info('secure2pc setting: server training DONE')

    logger.info('enhanced secure2pc setting: client start to encrypt dataset ...')
    ct_ff_lst_dict = dict()
    ct_bp_lst_dict = dict()
    x_idx_count = 0
    final_y_onehot_lst = None
    nn_client_enhanced_dict = dict()
    for id in setup_parties.keys():
        if x_idx_count == (len(setup_parties) - 1):
            n_features = X_data_lst[x_idx_count].shape[1] + 1
            nn_client_enhanced = CryptoNNClient(n_output=10, mini_batches=total_mini_batches, n_features=n_features,
                                                smc=es2pc_client, random_seed=520, id=id)
            nn_client_enhanced_dict[id] = nn_client_enhanced
            ct_feedforward_lst, ct_backpropagation_lst, y_onehot_lst = nn_client_enhanced.pre_process(X_data_lst[x_idx_count],
                                                                                             y_data)
            ct_ff_lst_dict[id] = ct_feedforward_lst
            ct_bp_lst_dict[id] = ct_backpropagation_lst
            final_y_onehot_lst = y_onehot_lst
        else:
            n_features = X_data_lst[x_idx_count].shape[1]
            nn_client_enhanced = CryptoNNClient(n_output=10, mini_batches=total_mini_batches, n_features=n_features,
                                                smc=es2pc_client, random_seed=520, id=id)
            nn_client_enhanced_dict[id] = nn_client_enhanced
            ct_feedforward_lst, ct_backpropagation_lst = nn_client_enhanced.pre_process(X_data_lst[x_idx_count])
            ct_ff_lst_dict[id] = ct_feedforward_lst
            ct_bp_lst_dict[id] = ct_backpropagation_lst
        x_idx_count = x_idx_count + 1
    logger.info('enhanced secure2pc setting: client encrypting DONE')
    for hidden_layers in hidden_layers_sets:
        nn_server_enhanced = CryptoNNServer(n_output=10, n_features=X_data.shape[1], hidden_layers=hidden_layers,
                                            l2=0.1, l1=0.0, epochs=1, eta=0.001, alpha=0.01,
                                            decrease_const=0.001, mini_batches=total_mini_batches, smc=es2pc_server)
        for id in nn_client_enhanced_dict.keys():
            nn_server_enhanced.register(nn_client_enhanced_dict[id])

        logger.info('enhanced secure2pc setting: server start to train ...')
        with timer('training using enhanced secure2pc setting - 5 batches - ' + str(len(hidden_layers)) + ' layers', logger) as t:
            nn_server_enhanced.fit((ct_ff_lst_dict, ct_bp_lst_dict), final_y_onehot_lst, X_test, y_test)
        logger.info('enhanced secure2pc setting: server training DONE')


def test_exp_nn_shallow_acc():
    logger.info('experiment for accuracy for full MNIST')

    logger.info('initialize the crypto system ...')
    sec_param_config_file = 'config/sec_param.json'  # indicate kernel size 5
    dlog_table_config_file = 'config/dlog_b8.json'
    with timer('initialize crypto system', logger) as t:
        eta = 1250
        sec_param = 256
        setup_parties = {
            'id_1': 200,
            'id_2': 200,
            'id_3': 200,
            'id_4': 200,
            'id_5': 200
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

    precision_data = 0
    precision_weight = 3

    s2pc_client = Secure2PCClient(sife=(sife_tpa, sife_enc_client), precision=precision_data)
    s2pc_server = Secure2PCServer(sife=(sife_tpa, sife_dec_client), precision=(precision_data, precision_weight))

    es2pc_client = EnhancedSecure2PCClient(
        sife=(sife_tpa, sife_enc_client),
        mife=(mife_tpa, mife_enc_client),
        precision=precision_data)
    es2pc_server = EnhancedSecure2PCServer(
        sife=(sife_tpa, sife_dec_client),
        mife=(mife_tpa, mife_dec_client),
        precision=(precision_data, precision_weight))

    # X_train, y_train = load_mnist_size('datasets/mnist', size=150)
    # X_test, y_test = load_mnist_size('datasets/mnist', size=100, kind='t10k')
    X_train, y_train = load_mnist('datasets/mnist')
    X_test, y_test = load_mnist('datasets/mnist', kind='t10k')

    # shuffle
    X_data, y_data = X_train.copy(), y_train.copy()
    idx = np.random.permutation(X_data.shape[0])
    X_data, y_data = X_data[idx], y_data[idx]

    total_mini_batches = 100
    epochs = 50

    nn_client_base = CryptoNNClient(n_output=10, mini_batches=total_mini_batches, n_features=X_data.shape[1], random_seed=520)
    nn_server_base = CryptoNNServer(n_output=10, n_features=X_data.shape[1], hidden_layers=[64],
                                    l2=0.1, l1=0.0, epochs=epochs, eta=0.001, alpha=0.001,
                                    decrease_const=0.0001, mini_batches=total_mini_batches)
    logger.info('base client start to pre-process ...')
    with timer('base client pre-process', logger) as t:
        X_client, y_client = nn_client_base.pre_process(X_data, y_data)
    logger.info('base client pre-process DONE')
    logger.info('base server start to train ...')
    (train_loss_hist_base,
     test_acc_hist_base,
     train_batch_time_hist_base,
     train_time_hist_base) = nn_server_base.fit(X_client, y_client, X_test, y_test)
    logger.info('base server training DONE')
    logger.info('train_loss_hist_base: \n\r' + str(train_loss_hist_base))
    logger.info('test_acc_hist_base: \n\r' + str(test_acc_hist_base))
    logger.info('train_batch_time_hist_base: \n\r' + str(train_batch_time_hist_base))
    logger.info('train_time_hist_base: \n\r' + str(train_time_hist_base))


    nn_client_s2pc = CryptoNNClient(n_output=10, mini_batches=total_mini_batches, n_features=X_data.shape[1],
                                    smc=s2pc_client, random_seed=520)
    nn_server_s2pc = CryptoNNServer(n_output=10, n_features=X_data.shape[1], hidden_layers=[64],
                                    l2=0.1, l1=0.0, epochs=epochs, eta=0.001, alpha=0.001,
                                    decrease_const=0.0001, mini_batches=total_mini_batches, smc=s2pc_server)
    logger.info('s2pc client start to pre-process - encrypt dataset ...')
    with timer('s2pc client pre-process', logger) as t:
        ct_feedforward_lst, ct_backpropagation_lst, y_onehot_lst = nn_client_s2pc.pre_process(X_data, y_data)
    logger.info('s2pc client pre-process DONE')
    logger.info('s2pc server start to train ...')
    (train_loss_hist_s2pc,
     test_acc_hist_s2pc,
     train_batch_time_hist_s2pc,
     train_time_hist_s2pc) = nn_server_s2pc.fit((ct_feedforward_lst, ct_backpropagation_lst), y_onehot_lst, X_test, y_test)
    logger.info('s2pc server training DONE')
    logger.info('train_loss_hist_s2pc: \n\r' + str(train_loss_hist_s2pc))
    logger.info('test_acc_hist_s2pc: \n\r' + str(test_acc_hist_s2pc))
    logger.info('train_batch_time_hist_s2pc: \n\r' + str(train_batch_time_hist_s2pc))
    logger.info('train_time_hist_s2pc: \n\r' + str(train_time_hist_s2pc))


    features_splits = np.array_split(range(X_data.shape[1]), len(setup_parties))
    X_data_lst = [X_data[:, idx] for idx in features_splits]
    nn_server = CryptoNNServer(n_output=10, n_features=X_data.shape[1], hidden_layers=[64],
                               l2=0.1, l1=0.0, epochs=epochs, eta=0.001, alpha=0.001,
                               decrease_const=0.0001, mini_batches=total_mini_batches, smc=es2pc_server)
    logger.info('es2pc client start to pre-process - encrypt dataset ...')
    ct_ff_lst_dict = dict()
    ct_bp_lst_dict = dict()
    x_idx_count = 0
    final_y_onehot_lst = None
    with timer('es2pc client pre-process', logger) as t:
        for id in setup_parties.keys():
            if x_idx_count == (len(setup_parties) - 1):
                n_features = X_data_lst[x_idx_count].shape[1] + 1
                nn_client = CryptoNNClient(n_output=10, mini_batches=total_mini_batches, n_features=n_features,
                                           smc=es2pc_client, random_seed=520, id=id)
                nn_server.register(nn_client)
                ct_feedforward_lst, ct_backpropagation_lst, y_onehot_lst = nn_client.pre_process(X_data_lst[x_idx_count],
                                                                                                 y_data)
                ct_ff_lst_dict[id] = ct_feedforward_lst
                ct_bp_lst_dict[id] = ct_backpropagation_lst
                final_y_onehot_lst = y_onehot_lst
            else:
                n_features = X_data_lst[x_idx_count].shape[1]
                nn_client = CryptoNNClient(n_output=10, mini_batches=total_mini_batches, n_features=n_features,
                                           smc=es2pc_client, random_seed=520, id=id)
                nn_server.register(nn_client)
                ct_feedforward_lst, ct_backpropagation_lst = nn_client.pre_process(X_data_lst[x_idx_count])
                ct_ff_lst_dict[id] = ct_feedforward_lst
                ct_bp_lst_dict[id] = ct_backpropagation_lst
            x_idx_count = x_idx_count + 1
    logger.info('es2pc client pre-processing DONE')
    logger.info('es2pc server start to train ...')
    (train_loss_hist_es2pc,
     test_acc_hist_es2pc,
     train_batch_time_hist_es2pc,
     train_time_hist_es2pc) = nn_server.fit((ct_ff_lst_dict, ct_bp_lst_dict), final_y_onehot_lst, X_test, y_test)
    logger.info('es2pc server training DONE')
    logger.info('train_loss_hist_es2pc: \n\r' + str(train_loss_hist_es2pc))
    logger.info('test_acc_hist_es2pc: \n\r' + str(test_acc_hist_es2pc))
    logger.info('train_batch_time_hist_es2pc: \n\r' + str(train_batch_time_hist_es2pc))
    logger.info('train_time_hist_es2pc: \n\r' + str(train_time_hist_es2pc))

def test_exp_nn_shallow_acc_simulate():
    logger.info('experiment for accuracy for full MNIST in simulation way')

    precision_weight_lst = [3, 5, 7, 9]

    # X_train, y_train = load_mnist_size('datasets/mnist', size=150)
    # X_test, y_test = load_mnist_size('datasets/mnist', size=100, kind='t10k')
    X_train, y_train = load_mnist('datasets/mnist')
    X_test, y_test = load_mnist('datasets/mnist', kind='t10k')

    # shuffle
    X_data, y_data = X_train.copy(), y_train.copy()
    idx = np.random.permutation(X_data.shape[0])
    X_data, y_data = X_data[idx], y_data[idx]

    total_mini_batches = 100
    epochs = 50
    test_acc_hist = list()

    for precision in precision_weight_lst:
        nn_client_base = CryptoNNClient(n_output=10, mini_batches=total_mini_batches, n_features=X_data.shape[1], random_seed=520)
        nn_server_base = CryptoNNServer(n_output=10, n_features=X_data.shape[1], hidden_layers=[512, 256, 128, 64, 32],
                                        l2=0.1, l1=0.0, epochs=epochs, eta=0.001, alpha=0.001,
                                        decrease_const=0.0001, mini_batches=total_mini_batches, precision=precision)
        logger.info('base client start to pre-process ...')
        with timer('base client pre-process', logger) as t:
            X_client, y_client = nn_client_base.pre_process(X_data, y_data)
        logger.info('base client pre-process DONE')
        logger.info('base server start to train ... with precision ' + str(precision))
        (train_loss_hist_base,
         test_acc_hist_base,
         train_batch_time_hist_base,
         train_time_hist_base) = nn_server_base.fit(X_client, y_client, X_test, y_test)
        logger.info('base server training DONE')
        test_acc_hist.append(test_acc_hist_base)

    logger.info('test_acc_hist_base: \n\r' + str(test_acc_hist))


def test_exp_nn_shallow_acc_cmp_simulate():
    logger.info('experiment for accuracy for full MNIST in simulation way')

    precision_weight_lst = [4, 5]

    # X_train, y_train = load_mnist_size('datasets/mnist', size=150)
    # X_test, y_test = load_mnist_size('datasets/mnist', size=100, kind='t10k')
    X_train, y_train = load_mnist('datasets/mnist')
    X_test, y_test = load_mnist('datasets/mnist', kind='t10k')

    # shuffle
    X_data, y_data = X_train.copy(), y_train.copy()
    idx = np.random.permutation(X_data.shape[0])
    X_data, y_data = X_data[idx], y_data[idx]

    total_mini_batches = 100
    epochs = 100
    test_acc_hist = list()

    for precision in precision_weight_lst:
        nn_client_base = CryptoNNClient(n_output=10, mini_batches=total_mini_batches, n_features=X_data.shape[1], random_seed=520)
        nn_server_base = CryptoNNServer(n_output=10, n_features=X_data.shape[1], hidden_layers=[512, 256, 128, 64, 32],
                                        l2=0.1, l1=0.0, epochs=epochs, eta=0.001, alpha=0.001,
                                        decrease_const=0.0001, mini_batches=total_mini_batches, precision=precision)
        logger.info('client start to pre-process ...')
        with timer('client pre-process', logger) as t:
            X_client, y_client = nn_client_base.pre_process(X_data, y_data)
        logger.info('client pre-process DONE')
        logger.info('server start to train ... with precision ' + str(precision))
        (train_loss_hist_base,
         test_acc_hist_base,
         train_batch_time_hist_base,
         train_time_hist_base) = nn_server_base.fit(X_client, y_client, X_test, y_test)
        logger.info('server training DONE')
        test_acc_hist.append(test_acc_hist_base)

    nn_client_base = CryptoNNClient(n_output=10, mini_batches=total_mini_batches, n_features=X_data.shape[1],
                                    random_seed=520)
    nn_server_base = CryptoNNServer(n_output=10, n_features=X_data.shape[1], hidden_layers=[512, 256, 128, 64, 32],
                                    l2=0.1, l1=0.0, epochs=epochs, eta=0.001, alpha=0.001,
                                    decrease_const=0.0001, mini_batches=total_mini_batches)
    logger.info('base client start to pre-process ...')
    with timer('base client pre-process', logger) as t:
        X_client, y_client = nn_client_base.pre_process(X_data, y_data)
    logger.info('base client pre-process DONE')
    logger.info('base server start to train ... ')
    (train_loss_hist_base,
     test_acc_hist_base,
     train_batch_time_hist_base,
     train_time_hist_base) = nn_server_base.fit(X_client, y_client, X_test, y_test)
    logger.info('base server training DONE')
    test_acc_hist.append(test_acc_hist_base)

    logger.info('test_acc_hist_base: \n\r' + str(test_acc_hist))





