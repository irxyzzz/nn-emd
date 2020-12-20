import abc
import six
import logging
import time

import numpy as np

from nn.smc import Secure2PCServer
from nn.smc import Secure2PCClient
from nn.smc import EnhancedSecure2PCServer
from nn.smc import EnhancedSecure2PCClient

logger = logging.getLogger(__name__)

@six.add_metaclass(abc.ABCMeta)
class NNEMDClient(object):
    def __init__(self, n_output, mini_batches, n_features, smc=None, random_seed=None, id=None):
        np.random.seed(random_seed)
        self.n_output = n_output
        self.mini_batches = mini_batches
        self.smc = smc
        self.id = id
        self.n_features = n_features

    def get_id(self):
        return self.id

    def get_features_size(self):
        return self.n_features

    def _add_bias_unit(self, X, how='column'):
        """Add bias unit (column or row of 1s) to array at index 0"""
        if how == 'column':
            X_new = np.ones((X.shape[0], X.shape[1] + 1))
            X_new[:, 1:] = X
        elif how == 'row':
            X_new = np.ones((X.shape[0] + 1, X.shape[1]))
            X_new[1:, :] = X
        else:
            raise AttributeError('`how` must be `column` or `row`')
        return X_new

    def _encode_labels(self, y, k):
        onehot = np.zeros((k, y.shape[0]))
        for idx, val in enumerate(y):
            onehot[val, idx] = 1.0
        return onehot

    def pre_process(self, X, y=None):
        X_data = X.copy()
        mini_batch = np.array_split(range(X_data.shape[0]), self.mini_batches)

        if y is not None:
            y_data = y.copy()
            y_onehot = self._encode_labels(y_data, self.n_output)
            y_onehot_lst = [y_onehot[:, idx] for idx in mini_batch]
            X_data = self._add_bias_unit(X_data, how='column') # let's suppose the party with label to setup the env.

        if self.smc:
            ct_feedforward_lst, ct_backpropagation_lst = None, None
            if isinstance(self.smc, Secure2PCClient):
                ct_feedforward_lst = [np.array(self.smc.execute_ndarray(X_data[idx])) for idx in mini_batch]
                ct_backpropagation_lst = [np.array(self.smc.execute_ndarray(X_data[idx].T)) for idx in mini_batch]
            elif isinstance(self.smc, EnhancedSecure2PCClient):
                ct_feedforward_lst = [np.array(self.smc.execute_ndarray(X_data[idx], {'type': 'mife', 'id': self.id})) for idx in mini_batch]
                ct_backpropagation_lst = [np.array(self.smc.execute_ndarray(X_data[idx].T, {'type': 'sife'})) for idx in mini_batch]
            if y is not None:
                return ct_feedforward_lst, ct_backpropagation_lst, y_onehot_lst
            else:
                return ct_feedforward_lst, ct_backpropagation_lst
        else:
            X_data_lst = [X_data[idx] for idx in mini_batch]
            if y is not None:
                return X_data_lst, y_onehot_lst
            else:
                return X_data_lst


@six.add_metaclass(abc.ABCMeta)
class NNEMDServer(object):
    def __init__(self, n_output, n_features, hidden_layers,
                 l1=0.0, l2=0.0, epochs=500, eta=0.001,
                 alpha=0.0, decrease_const=0.0,
                 mini_batches=1, smc=None, precision=None):
        self.n_output = n_output
        self.n_features = n_features
        self.hidden_layers = hidden_layers
        self.w = self._initialize_weights()
        self.l1 = l1
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.alpha = alpha
        self.decrease_const = decrease_const
        self.mini_batches = mini_batches
        self.smc = smc
        self.precision = precision
        self.parties = dict()

    def register(self, party):
        self.parties[party.get_id()] = party.get_features_size()

    def _initialize_weights(self):
        self.layers = [self.n_features] + self.hidden_layers + [self.n_output]
        w = [np.random.uniform(-1.0, 1.0, (self.layers[i+1], self.layers[i] + 1)) for i in range(len(self.layers) - 1)]
        return w

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def _sigmoid_gradient(self, z):
        sg = self._sigmoid(z)
        return sg * (1.0 - sg)

    def _add_bias_unit(self, X, how='column'):
        """Add bias unit (column or row of 1s) to array at index 0"""
        if how == 'column':
            X_new = np.ones((X.shape[0], X.shape[1] + 1))
            X_new[:, 1:] = X
        elif how == 'row':
            X_new = np.ones((X.shape[0] + 1, X.shape[1]))
            X_new[1:, :] = X
        else:
            raise AttributeError('`how` must be `column` or `row`')
        return X_new

    def _feedforward(self, X, w):
        z = [None for i in range(len(w))]
        a = [None for i in range(len(w))]
        if self.precision:
            z[0] = (w[0]*pow(10, self.precision)).astype(int).dot(X.T) / pow(10, self.precision)
        else:
            z[0] = w[0].dot(X.T)
        a[0] = self._add_bias_unit(self._sigmoid(z[0]), how='row')
        for i in range(1, len(w)):
            z[i] = w[i].dot(a[i-1])
            if i != len(w) - 1:
                a[i] = self._add_bias_unit(self._sigmoid(z[i]), how='row')
            else:
                a[i] = self._sigmoid(z[i])
        return z, a

    def _feedforward_secure(self, ct_batch, w):
        z = [None for i in range(len(w))]
        a = [None for i in range(len(w))]

        # simulate the smc
        if isinstance(self.smc, Secure2PCServer):
            sk_w0 = self.smc.request_key_ndarray(w[0])
            z[0] = self.smc.execute_ndarray(sk_w0, ct_batch.tolist(), w[0])
        elif isinstance(self.smc, EnhancedSecure2PCServer):
            sk_w0 = self.smc.request_key_ndarray(w[0], {'type': 'mife', 'parties': self.parties})
            z[0] = self.smc.execute_ndarray(sk_w0, ct_batch, w[0], {'type': 'mife'})
        # end of smc
        a[0] = self._add_bias_unit(self._sigmoid(z[0]), how='row')

        for i in range(1, len(w)):
            z[i] = w[i].dot(a[i - 1])
            if i != len(w) - 1:
                a[i] = self._add_bias_unit(self._sigmoid(z[i]), how='row')
            else:
                a[i] = self._sigmoid(z[i])
        return z, a

    def _L2_reg(self, lambda_, w):
        res = 0.0
        for i in range(len(w)):
            res += np.sum(w[i][:, 1:] ** 2)
        return (lambda_/2.0) * res

    def _L1_reg(self, lambda_, w):
        res = 0.0
        for i in range(len(w)):
            res += np.abs(w[i][:, 1:]).sum()
        return (lambda_/2.0) * res

    def _get_cost(self, y_encode, output, w):
        term1 = - y_encode * (np.log(output))
        term2 = (1.0 - y_encode) * np.log(1.0 - output)
        cost = np.sum(term1 - term2)
        L1_term = self._L1_reg(self.l1, w)
        L2_term = self._L2_reg(self.l2, w)
        cost = cost + L1_term + L2_term
        return cost

    def compute_cost(self, AL, Y):
        m = Y.shape[1]
        # Compute loss from aL and y.
        cost = (1. / m) * (-np.dot(Y, np.log(AL).T) - np.dot(1 - Y, np.log(1 - AL).T))
        cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
        assert (cost.shape == ())
        return cost

    def _get_gradient(self, x, y_encode, a, z, w):
        # back-propagation
        sigma = [None for i in range(len(w))]
        grad = [None for i in range(len(w))]
        sigma[-1] = a[-1] - y_encode
        for i in range(len(w)-2, -1, -1):
            sigma[i] = w[i+1].T.dot(sigma[i+1]) * self._sigmoid_gradient(self._add_bias_unit(z[i], how='row'))
            sigma[i] = sigma[i][1:, :]
        if self.precision:
            grad[0] = (sigma[0]*pow(10, self.precision)).astype(int).dot(x) / pow(10, self.precision)
        else:
            grad[0] = sigma[0].dot(x)
        for i in range(1, len(w)):
            grad[i] = sigma[i].dot(a[i-1].T)

        # regularize
        for i in range(len(w)):
            grad[i][:, 1:] += self.l2 * w[i][:, 1:]
            grad[i][:, 1:] += self.l1 * np.sign(w[i][:, 1:])

        return grad

    def _get_gradient_secure(self, ct_batch, y_encode, a, z, w):
        sigma = [None for i in range(len(w))]
        grad = [None for i in range(len(w))]
        sigma[-1] = a[-1] - y_encode
        for i in range(len(w) - 2, -1, -1):
            sigma[i] = w[i + 1].T.dot(sigma[i + 1]) * self._sigmoid_gradient(self._add_bias_unit(z[i], how='row'))
            sigma[i] = sigma[i][1:, :]

        # using smc
        if isinstance(self.smc, Secure2PCServer):
            sk_sigma0 = self.smc.request_key_ndarray(sigma[0])
            grad[0] = self.smc.execute_ndarray(sk_sigma0, ct_batch.tolist(), sigma[0])
        elif isinstance(self.smc, EnhancedSecure2PCServer):
            sk_sigma0 = self.smc.request_key_ndarray(sigma[0], {'type': 'sife'})
            grad[0] = self.smc.execute_ndarray(sk_sigma0, ct_batch, sigma[0], {'type': 'sife'})
        # end of smc

        for i in range(1, len(w)):
            grad[i] = sigma[i].dot(a[i - 1].T)

        # regularize
        for i in range(len(w)):
            grad[i][:, 1:] += self.l2 * w[i][:, 1:]
            grad[i][:, 1:] += self.l1 * np.sign(w[i][:, 1:])

        return grad

    def predict(self, X):
        if len(X.shape) != 2:
            raise AttributeError('X must be a [n_samples, n_features] array.\n'
                                 'Use X[:,None] for 1-feature classification,'
                                 '\nor X[[i]] for 1-sample classification')
        X = self._add_bias_unit(X, how='column')
        z, a = self._feedforward(X, self.w)
        y_pred = np.argmax(z[len(z)-1], axis=0)
        return y_pred

    def fit(self, x, y, x_test, y_test, print_progress=False):
        train_loss_hist = list()
        test_acc_hist = list()
        train_time_hist = list()
        train_batch_time_hist = list()

        delta_w_prev = [np.zeros(self.w[i].shape) for i in range(len(self.w))]

        start_time = time.perf_counter()
        for epoch in range(self.epochs):
            logger.info('Epoch: %d/%d start ... ' % (epoch + 1, self.epochs))
            if print_progress:
                print('Epoch: %d/%d' % (epoch+1, self.epochs))
            # adaptive learning rate
            self.eta /= (1 + self.decrease_const*epoch)

            cost = 0.0
            for i in range(len(y)):
                # feedforward
                if self.smc:
                    if isinstance(self.smc, Secure2PCServer):
                        z, a = self._feedforward_secure(x[0][i], self.w)
                        cost += self._get_cost(y_encode=y[i], output=a[-1], w=self.w)
                        grad = self._get_gradient_secure(ct_batch=x[1][i], y_encode=y[i], a=a, z=z, w=self.w)
                    elif isinstance(self.smc, EnhancedSecure2PCServer):
                        # fuse the mife cts of one batch from multiple parties
                        batch_size = y[i].shape[1]
                        ct_ff_batch_lst = [None for b_idx in range(batch_size)]
                        for b_idx in range(batch_size):
                            ct_sample = dict()
                            ct_sample['parties'] = self.parties
                            ct_sample['ct_dict'] = dict()
                            for id in self.parties.keys():
                                ct_sample['ct_dict'][id] = x[0][id][i][b_idx]
                            ct_ff_batch_lst[b_idx] = ct_sample

                        ct_bp_batch_lst = list()
                        for id in self.parties.keys():
                            ct_bp_batch_lst = ct_bp_batch_lst + x[1][id][i].tolist()

                        z, a = self._feedforward_secure(ct_ff_batch_lst, self.w)
                        cost += self._get_cost(y_encode=y[i], output=a[-1], w=self.w)
                        grad = self._get_gradient_secure(ct_batch=ct_bp_batch_lst, y_encode=y[i], a=a, z=z, w=self.w)
                else:
                    z, a = self._feedforward(x[i], self.w)
                    cost += self._get_cost(y_encode=y[i], output=a[-1], w=self.w)
                    grad = self._get_gradient(x=x[i], y_encode=y[i], a=a, z=z, w=self.w)

                delta_w = [self.eta * grad[i] for i in range(len(self.w))]
                for i in range(len(self.w)):
                    self.w[i] -= (delta_w[i] + (self.alpha * delta_w_prev[i]))
                delta_w_prev = delta_w

                train_batch_time_hist.append(time.perf_counter() - start_time)

            y_test_pred = self.predict(x_test)
            test_acc = np.sum(y_test == y_test_pred, axis=0) / x_test.shape[0]
            test_acc_hist.append(test_acc)
            train_loss_hist.append(cost/len(y))
            train_time_hist.append(time.perf_counter() - start_time)
            logger.info('Epoch: %d/%d done - train loss %.2f - test acc %.2f%%' % (epoch + 1, self.epochs, cost/len(y), test_acc * 100))
        return train_loss_hist, test_acc_hist, train_batch_time_hist, train_time_hist