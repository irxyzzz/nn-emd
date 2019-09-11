import sys
import datetime
import logging

import numpy as np
import matplotlib.pyplot as plt

from nn.shallow.nn_shallow import NNShallow
from nn.utils import load_mnist_size
from nn.utils import timer

t_str = str(datetime.datetime.today())
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
    datefmt='%m-%d %H:%M',
    filename="logs/test_nn_shallow-" + '-'.join(t_str.split()[:1] + t_str.split()[1].split(':')[:2]) + '.log',
    filemode='w')
logger = logging.getLogger(__name__)


def labels_mapping(y, r):
    y_prime = (y + r) % (np.unique(y).shape[0])
    return y_prime


def test_nn_shallow_mnist():
    X_train, y_train = load_mnist_size('datasets/mnist', size=600)
    # y_train = labels_mapping(y_train, 7)
    X_test, y_test = load_mnist_size('datasets/mnist', size=100, kind='t10k')
    # y_test = labels_mapping(y_test, 7)
    # X_train, y_train = load_mnist('datasets/mnist')
    # X_test, y_test = load_mnist('datasets/mnist', kind='t10k')
    nn = NNShallow(n_output=10,
                   n_features=X_train.shape[1],
                   n_hidden=50,
                   l2=0.1,
                   l1=0.0,
                   epochs=1000,
                   eta=0.001,
                   alpha=0.001,
                   decrease_const=0.00001,
                   minibatches=50,
                   shuffle=True,
                   random_state=520)
    nn.fit(X_train, y_train)

    # plt.plot(range(len(nn.cost_)), nn.cost_)
    # plt.ylim([0, 2000])
    # plt.ylabel('Cost')
    # plt.xlabel('Epochs * 50')
    # plt.tight_layout()
    # # plt.savefig('./figures/cost.png', dpi=300)
    # plt.show()
    #
    # batches = np.array_split(range(len(nn.cost_)), 1000)
    # cost_ary = np.array(nn.cost_)
    # cost_avgs = [np.mean(cost_ary[i]) for i in batches]
    # plt.plot(range(len(cost_avgs)), cost_avgs, color='red')
    # plt.ylim([0, 2000])
    # plt.ylabel('Cost')
    # plt.xlabel('Epochs')
    # plt.tight_layout()
    # # plt.savefig('./figures/cost2.png', dpi=300)
    # plt.show()

    y_train_pred = nn.predict(X_train)
    acc = np.sum(y_train == y_train_pred, axis=0) / X_train.shape[0]
    print('Training accuracy: %.2f%%' % (acc * 100))

    y_test_pred = nn.predict(X_test)
    acc = np.sum(y_test == y_test_pred, axis=0) / X_test.shape[0]
    print('Test accuracy: %.2f%%' % (acc * 100))