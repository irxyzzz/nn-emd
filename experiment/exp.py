import datetime
import logging
import os
import numpy as np
import matplotlib.pyplot as plt

crypto_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.insert(0, crypto_path)

from crypto.sife import SIFE
from nn.utils import load_mnist
from nn.utils import load_mnist_size
from nn.utils import timer
from nn.cnn.lenet5 import LeNet5
from nn.smc import Secure2PC

t_str = str(datetime.datetime.today())
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
    datefmt='%m-%d %H:%M',
    filename="../logs/" + __name__ + '-' + '-'.join(t_str.split()[:1] + t_str.split()[1].split(':')[:2]) + '.log',
    filemode='w')

logger = logging.getLogger(__name__)


def exp_cmp_lenet5_smc():
    logger.info('Loading the dataset ...')

    batch_size = 480
    epoch = 2

    train_images, train_labels = load_mnist('../datasets/mnist')
    test_images, test_labels = load_mnist('../datasets/mnist', kind='t10k')
    # size = batch_size * 300
    # train_images, train_labels = load_mnist_size('../datasets/mnist/', size=size, kind='train')
    # test_images, test_labels = load_mnist_size('../datasets/mnist/', size=size, kind='t10k')

    logger.info('Pre-processing data ...')
    num_classes = 10
    train_images -= int(np.mean(train_images))
    train_images /= int(np.std(train_images))
    test_images -= int(np.mean(test_images))
    test_images /= int(np.std(test_images))
    training_data = train_images.reshape(60000, 1, 28, 28)
    training_labels = np.eye(num_classes)[train_labels]
    testing_data = test_images.reshape(10000, 1, 28, 28)
    testing_labels = np.eye(num_classes)[test_labels]

    lenet5_normal = LeNet5(lr=0.01)
    logger.info('Start to train normal LeNet-5')
    (train_loss_hist_base,
     train_total_acc_hist_base,
     train_batch_acc_hist_base,
     test_acc_hist_base,
     train_time_hist_base) = lenet5_normal.train_and_eval(training_data, training_labels, batch_size, epoch,
                                                          testing_data, testing_labels)
    logger.info('base - train loss hist: \n' + str(train_loss_hist_base))
    logger.info('base - train total acc hist: \n' + str(train_total_acc_hist_base))
    logger.info('base - train batch acc hist: \n' + str(train_batch_acc_hist_base))
    logger.info('base - train cost time hist: \n' + str(train_time_hist_base))
    logger.info('base - test acc hist: \n' + str(test_acc_hist_base))

    logger.info('Initialize the crypto system ...')
    tpa_config_file = '../config/sife_v25_b8.json'  # indicate kernel size 5
    client_config_file = '../config/sife_v25_b8_dlog.json'
    with timer('Load sife config file, cost time', logger) as t:
        sife = SIFE(tpa_config=tpa_config_file, client_config=client_config_file)
    smc = Secure2PC(crypto=sife, vec_len=25, precision=3)

    lenet5_smc = LeNet5(lr=0.01, smc=smc)
    logger.info('Start to train crypto LeNet-5')
    (train_loss_hist_smc,
     train_total_acc_hist_smc,
     train_batch_acc_hist_smc,
     test_acc_hist_smc,
     train_time_hist_smc) = lenet5_smc.train_and_eval(training_data, training_labels, batch_size, epoch,
                                                       testing_data, testing_labels)

    logger.info('cryptonn - train loss hist: \n' + str(train_loss_hist_smc))
    logger.info('cryptonn - train total acc hist: \n' + str(train_total_acc_hist_smc))
    logger.info('cryptonn - train batch acc hist: \n' + str(train_batch_acc_hist_smc))
    logger.info('cryptonn - train cost time hist: \n' + str(train_time_hist_smc))
    logger.info('cryptonn - test acc hist: \n' + str(test_acc_hist_smc))

    suffix = '-'.join(str(datetime.datetime.now()).split()[:1]
                      + str(datetime.datetime.now()).split()[1].split(':')[:2])

    plt.plot(train_loss_hist_base)
    plt.plot(train_loss_hist_smc)
    plt.xlabel("#iteration-batches")
    plt.ylabel("train batch loss")
    plt.legend(("normal", "cryptonn"))
    plt.savefig('res/cmp_lenet5_train_loss_' + suffix + '.eps')
    # plt.show()
    plt.close()

    plt.plot(train_time_hist_base)
    plt.plot(train_time_hist_smc)
    plt.xlabel("#iteration-batches")
    plt.ylabel("training time cost")
    plt.legend(("normal", "cryptonn"))
    plt.savefig('res/cmp_lenet5_train_cost_time_' + suffix + '.eps')
    # plt.show()
    plt.close()

    plt.plot(train_total_acc_hist_base)
    plt.plot(train_batch_acc_hist_base)
    plt.plot(test_acc_hist_base)
    plt.plot(train_total_acc_hist_smc)
    plt.plot(train_batch_acc_hist_smc)
    plt.plot(test_acc_hist_smc)
    plt.xlabel("#iteration-batches")
    plt.ylabel("accuracy")
    plt.legend(('normal - train total acc',
                'normal - train batch acc',
                'normal - test acc',
                'cryptonn - train total acc',
                'cryptonn - train batch acc',
                'cryptonn - test acc'))
    plt.savefig('res/cmp_lenet5_acc_' + suffix + '.eps')
    # plt.show()


if __name__ == '__main__':
    exp_cmp_lenet5_smc()