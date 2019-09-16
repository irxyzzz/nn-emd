import datetime
import logging
import numpy as np
import matplotlib.pyplot as plt


from crypto.sife import SIFE
from nn.utils import load_mnist
from nn.utils import load_mnist_size
from nn.utils import timer
from nn.cnn.lenet5 import LeNet5

t_str = str(datetime.datetime.today())
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
    datefmt='%m-%d %H:%M',
    filename="logs/" + __name__ + '-' + '-'.join(t_str.split()[:1] + t_str.split()[1].split(':')[:2]) + '.log',
    filemode='w')

logger = logging.getLogger(__name__)

np.random.seed(520)

def test_lenet5_with_output():
    logger.info('Loading the dataset ...')

    batch_size = 32
    epoch = 1
    size = batch_size * 30

    train_images, train_labels = load_mnist('datasets/mnist')
    test_images, test_labels = load_mnist('datasets/mnist', kind='t10k')
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

    lenet5 = LeNet5(lr=0.01)
    logger.info('Start to train LeNet-5')
    (train_loss_hist,
     train_total_acc_hist,
     train_batch_acc_hist,
     test_acc_hist,
     train_time_hist) = lenet5.train_and_eval(training_data,
                                            training_labels,
                                            batch_size,
                                            epoch,
                                            testing_data,
                                            testing_labels)
    # plt.plot(train_loss_hist)
    # plt.xlabel("#iteration-batches")
    # plt.ylabel("batch loss")
    # plt.legend(("train batch loss"))
    # plt.savefig('lenet5_train_loss.eps')
    # # plt.show()
    # plt.close()

    plt.plot(train_total_acc_hist)
    plt.plot(train_batch_acc_hist)
    plt.plot(test_acc_hist)
    plt.xlabel("#iteration-batches")
    plt.ylabel("accuracy")
    plt.legend(('train total acc', 'train batch acc', 'test acc'))
    # plt.savefig('lenet5_acc.eps')
    plt.show()