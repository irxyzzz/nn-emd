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
from crypto.sife_dynamic import SIFEDynamicTPA
from crypto.sife_dynamic import SIFEDynamicClient
from crypto.mife_dynamic import MIFEDynamicTPA
from crypto.mife_dynamic import MIFEDynamicClient

t_str = str(datetime.datetime.today())
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
    datefmt='%m-%d %H:%M',
    filename="logs/test_nn_shallow_cs-" + '-'.join(t_str.split()[:1] + t_str.split()[1].split(':')[:2]) + '.log',
    filemode='w')
logger = logging.getLogger(__name__)


def test_nn_shallow_mnist():
    # X_train, y_train = load_mnist_size('datasets/mnist', size=600)
    # X_test, y_test = load_mnist_size('datasets/mnist', size=100, kind='t10k')
    X_train, y_train = load_mnist('datasets/mnist')
    X_test, y_test = load_mnist('datasets/mnist', kind='t10k')

    # shuffle
    X_data, y_data = X_train.copy(), y_train.copy()
    idx = np.random.permutation(X_data.shape[0])
    X_data, y_data = X_data[idx], y_data[idx]

    total_mini_batches = 100

    nn_client = CryptoNNClient(n_output=10, mini_batches=total_mini_batches, n_features=X_data.shape[1], random_seed=520)
    nn_server = CryptoNNServer(n_output=10, n_features=X_data.shape[1], n_hidden=[128, 64],
                               l2=0.01, l1=0.0, epochs=20, eta=0.001, alpha=0.01,
                               decrease_const=0.001, mini_batches=total_mini_batches)

    X_client, y_client = nn_client.pre_process(X_data, y_data)
    (train_loss_hist,
     test_acc_hist,
     train_batch_time_hist,
     train_time_hist) = nn_server.fit(X_client, y_client, X_test, y_test)
    logger.info('train loss: \n' + str(train_loss_hist))
    logger.info('test acc: \n' + str(test_acc_hist))

def test_nn_shallow_mnist_smc():
    logger.info('test nn shallow mnist with secure 2pc setting')
    logger.info('initialize the crypto system ...')
    sec_param_config_file = 'config/sec_param.json'  # indicate kernel size 5
    dlog_table_config_file = 'config/dlog_b8.json'
    with timer('load sife config file, cost time', logger) as t:
        eta = 1250
        sec_param = 256
        dlog = load_dlog_table_config(dlog_table_config_file)

        sife_tpa = SIFEDynamicTPA(eta, sec_param=sec_param, sec_param_config=sec_param_config_file)
        sife_tpa.setup()
        sife_enc_client = SIFEDynamicClient(role='enc')
        sife_dec_client = SIFEDynamicClient(role='dec', dlog=dlog)
        logger.info('the crypto system initialization done!')

    precision_data = 0
    precision_weight = 3

    secure2pc_client = Secure2PCClient(sife=(sife_tpa, sife_enc_client), precision=precision_data)
    secure2pc_server = Secure2PCServer(sife=(sife_tpa, sife_dec_client), precision=(precision_data, precision_weight))

    X_train, y_train = load_mnist_size('datasets/mnist', size=600)
    X_test, y_test = load_mnist_size('datasets/mnist', size=100, kind='t10k')
    # X_train, y_train = load_mnist('datasets/mnist')
    # X_test, y_test = load_mnist('datasets/mnist', kind='t10k')

    # shuffle
    X_data, y_data = X_train.copy(), y_train.copy()
    idx = np.random.permutation(X_data.shape[0])
    X_data, y_data = X_data[idx], y_data[idx]

    total_mini_batches = 10

    nn_client = CryptoNNClient(n_output=10, mini_batches=total_mini_batches, n_features=X_data.shape[1], smc=secure2pc_client, random_seed=520)
    nn_server = CryptoNNServer(n_output=10, n_features=X_data.shape[1], n_hidden=[128, 32],
                               l2=0.1, l1=0.0, epochs=2, eta=0.001, alpha=0.001,
                               decrease_const=0.001, mini_batches=total_mini_batches, smc=secure2pc_server)
    logger.info('client start to encrypt dataset ...')
    ct_feedforward_lst, ct_backpropagation_lst, y_onehot_lst = nn_client.pre_process(X_data, y_data)
    logger.info('client encrypting DONE')
    logger.info('server start to train ...')
    with timer('training using secure2pc setting - 10 batches', logger) as t:
        (train_loss_hist,
         test_acc_hist,
        train_batch_time_hist,
        train_time_hist) = nn_server.fit((ct_feedforward_lst, ct_backpropagation_lst), y_onehot_lst, X_test, y_test)
    logger.info('server training DONE')

    logger.info('training loss: \n\r' + str(train_loss_hist))
    logger.info('test acc: \n\r' + str(test_acc_hist))

def test_nn_shallow_mnist_smc_enhanced():
    logger.info('test nn shallow in mnist using enhanced smc')

    logger.info('initialize the crypto system ...')
    sec_param_config_file = 'config/sec_param.json'  # indicate kernel size 5
    dlog_table_config_file = 'config/dlog_b8.json'
    with timer('initialize crypto system, cost time', logger) as t:
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
    # X_train, y_train = load_mnist('datasets/mnist')
    # X_test, y_test = load_mnist('datasets/mnist', kind='t10k')

    # shuffle
    X_data, y_data = X_train.copy(), y_train.copy()
    idx = np.random.permutation(X_data.shape[0])
    X_data, y_data = X_data[idx], y_data[idx]

    features_splits = np.array_split(range(X_data.shape[1]), len(setup_parties))
    X_data_lst = [X_data[:, idx] for idx in features_splits]

    total_mini_batches = 10

    nn_server = CryptoNNServer(n_output=10, n_features=X_data.shape[1], n_hidden=[128, 32],
                               l2=0.1, l1=0.0, epochs=2, eta=0.001, alpha=0.001,
                               decrease_const=0.001, mini_batches=total_mini_batches, smc=es2pc_server)
    logger.info('client start to encrypt dataset ...')
    ct_ff_lst_dict = dict()
    ct_bp_lst_dict = dict()
    x_idx_count = 0
    final_y_onehot_lst = None
    for id in setup_parties.keys():
        if x_idx_count == (len(setup_parties) - 1):
            n_features = X_data_lst[x_idx_count].shape[1] + 1
            nn_client = CryptoNNClient(n_output=10, mini_batches=total_mini_batches, n_features=n_features,
                                       smc=es2pc_client, random_seed=520, id=id)
            nn_server.register(nn_client)
            ct_feedforward_lst, ct_backpropagation_lst, y_onehot_lst = nn_client.pre_process(X_data_lst[x_idx_count], y_data)
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
    logger.info('client encrypting DONE')

    logger.info('server start to train ...')
    (train_loss_hist,
     test_acc_hist,
     train_batch_time_hist,
     train_time_hist) = nn_server.fit((ct_ff_lst_dict, ct_bp_lst_dict), final_y_onehot_lst, X_test, y_test)
    logger.info('server training DONE')

    logger.info('training loss: \n\r' + str(train_loss_hist))
    logger.info('test acc: \n\r' + str(test_acc_hist))

def test_nn_shallow_mnist_smc_cryptonn():
    logger.info('test nn shallow mnist with secure 2pc setting')
    logger.info('initialize the crypto system ...')

    eta = 1250
    sec_param = 256

    sife_tpa = SIFEDynamicTPA(eta, sec_param=sec_param)
    sife_tpa.setup()
    sife_enc_client = SIFEDynamicClient(role='enc')
    sife_dec_client = SIFEDynamicClient(role='dec')
    logger.info('the crypto system initialization done!')

    precision_data = 0
    precision_weight = 4

    secure2pc_client = Secure2PCClient(sife=(sife_tpa, sife_enc_client), precision=precision_data)
    secure2pc_server = Secure2PCServer(sife=(sife_tpa, sife_dec_client), precision=(precision_data, precision_weight))

    X_train, y_train = load_mnist_size('datasets/mnist', size=60)
    X_test, y_test = load_mnist_size('datasets/mnist', size=100, kind='t10k')
    # X_train, y_train = load_mnist('datasets/mnist')
    # X_test, y_test = load_mnist('datasets/mnist', kind='t10k')

    # shuffle
    X_data, y_data = X_train.copy(), y_train.copy()
    idx = np.random.permutation(X_data.shape[0])
    X_data, y_data = X_data[idx], y_data[idx]

    total_mini_batches = 1

    nn_client = CryptoNNClient(n_output=10, mini_batches=total_mini_batches, n_features=X_data.shape[1], smc=secure2pc_client, random_seed=520)
    nn_server = CryptoNNServer(n_output=10, n_features=X_data.shape[1], n_hidden=[128, 32],
                               l2=0.1, l1=0.0, epochs=1, eta=0.001, alpha=0.001,
                               decrease_const=0.001, mini_batches=total_mini_batches, smc=secure2pc_server)
    logger.info('client start to encrypt dataset ...')
    ct_feedforward_lst, ct_backpropagation_lst, y_onehot_lst = nn_client.pre_process(X_data, y_data)
    logger.info('client encrypting DONE')
    logger.info('server start to train ...')
    with timer('training using secure2pc setting - 1 batches', logger) as t:
        (train_loss_hist,
         test_acc_hist,
         train_batch_time_hist,
         train_time_hist) = nn_server.fit((ct_feedforward_lst, ct_backpropagation_lst), y_onehot_lst, X_test, y_test)
    logger.info('server training DONE')

    logger.info('training loss: \n\r' + str(train_loss_hist))
    logger.info('test acc: \n\r' + str(test_acc_hist))