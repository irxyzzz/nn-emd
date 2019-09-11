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
from crypto.sife_dynamic import SIFEDynamicTPA
from crypto.sife_dynamic import SIFEDynamicClient

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

    nn_client = CryptoNNClient(n_output=10, mini_batches=50, shuffle=True, random_seed=520)
    nn_server = CryptoNNServer(n_output=10, n_features=X_train.shape[1], n_hidden=200,
                               l2=0.1, l1=0.0, epochs=10, eta=0.001, alpha=0.001,
                               decrease_const=0.00001, mini_batches=50)

    X_client, y_client = nn_client.pre_process(X_train, y_train)
    train_lost_hist = nn_server.fit(X_client, y_client, print_progress=False)
    print(train_lost_hist)

    y_train_pred = nn_server.predict(X_train)
    train_acc = np.sum(y_train == y_train_pred, axis=0) / X_train.shape[0]
    print('Training accuracy: %.2f%%' % (train_acc * 100))

    y_test_pred = nn_server.predict(X_test)
    test_acc = np.sum(y_test == y_test_pred, axis=0) / X_test.shape[0]
    print('Test accuracy: %.2f%%' % (test_acc * 100))


def test_nn_shallow_mnist_smc():
    logger.info('initialize the crypto system ...')
    sec_param_config_file = 'config/sec_param.json'  # indicate kernel size 5
    dlog_table_config_file = 'config/dlog_b8.json'
    with timer('load sife config file, cost time', logger) as t:
        eta = 1250
        sec_param = 256
        sife_tpa = SIFEDynamicTPA(eta, sec_param=sec_param, sec_param_config=sec_param_config_file)
        sife_tpa.setup()
        sife_enc_client = SIFEDynamicClient(role='enc')
        sife_dec_client = SIFEDynamicClient(role='dec', dlog_table_config=dlog_table_config_file)
        logger.info('the crypto system initialization done!')

    precision_data = 0
    precision_weight = 3

    secure2pc_client = Secure2PCClient(crypto=(sife_tpa, sife_enc_client), precision=precision_data)
    secure2pc_server = Secure2PCServer(crypto=(sife_tpa, sife_dec_client), precision=(precision_data, precision_weight))

    # X_train, y_train = load_mnist_size('datasets/mnist', size=600)
    # X_test, y_test = load_mnist_size('datasets/mnist', size=100, kind='t10k')
    X_train, y_train = load_mnist('datasets/mnist')
    X_test, y_test = load_mnist('datasets/mnist', kind='t10k')

    nn_client = CryptoNNClient(n_output=10, mini_batches=50, shuffle=True, smc=secure2pc_client, random_seed=520)
    nn_server = CryptoNNServer(n_output=10, n_features=X_train.shape[1], n_hidden=50,
                               l2=0.1, l1=0.0, epochs=2, eta=0.001, alpha=0.001,
                               decrease_const=0.00001, mini_batches=50, smc=secure2pc_server)
    logger.info('client start to encrypt dataset ...')
    ct_feedforward_lst, ct_backpropagation_lst, y_onehot_lst = nn_client.pre_process(X_train, y_train)
    logger.info('client encrypting DONE')
    logger.info('server start to train ...')
    train_lost_hist = nn_server.fit((ct_feedforward_lst, ct_backpropagation_lst), y_onehot_lst, print_progress=True)
    logger.info('server training DONE')

    y_train_pred = nn_server.predict(X_train)
    train_acc = np.sum(y_train == y_train_pred, axis=0) / X_train.shape[0]
    print('Training accuracy: %.2f%%' % (train_acc * 100))
    logging.info('Training accuracy: %.2f%%' % (train_acc * 100))

    y_test_pred = nn_server.predict(X_test)
    test_acc = np.sum(y_test == y_test_pred, axis=0) / X_test.shape[0]
    print('Test accuracy: %.2f%%' % (test_acc * 100))
    logging.info('Test accuracy: %.2f%%' % (test_acc * 100))