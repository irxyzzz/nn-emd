from nn.utils_nn_mnist import load_mnist, load_mnist_size
import matplotlib.pyplot as plt


def test_load_mnist():
    X_train, y_train = load_mnist('../nn/datasets/', kind='train')
    print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))
    # print(X_train[0])
    # print(X_train[9].reshape(28, 28))
    print(y_train.shape)
    print(X_train.shape)
    X_test, y_test = load_mnist('../nn/datasets/', kind='t10k')
    print('Rows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))


def test_load_mnist_size():
    X_train, y_train = load_mnist_size('../nn/datasets/', size=1000, kind='train')
    print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))
    # print(X_train[0])
    # print(X_train[9].reshape(28, 28))
    print(y_train.shape)
    print(X_train.shape)
    X_test, y_test = load_mnist('../nn/datasets/', kind='t10k')
    print('Rows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))


def check_image():
    X_train, y_train = load_mnist('../nn/datasets/', kind='train')
    print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))
    X_test, y_test = load_mnist('../nn/datasets/', kind='t10k')
    print('Rows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))
    # check the figure
    fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True, )
    ax = ax.flatten()
    for i in range(10):
        # tmp = X_train[y_train == i][0]
        img = X_train[y_train == i][0].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    # plt.savefig('./figures/mnist_all.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    # test_load_mnist()
    test_load_mnist_size()
    # check_image()