import abc
import six
import logging

import numpy as np

from nn.smc import Secure2PCServer
from nn.smc import Secure2PCClient
from nn.smc import EnhancedSecure2PCServer
from nn.smc import EnhancedSecure2PCClient

logger = logging.getLogger(__name__)

@six.add_metaclass(abc.ABCMeta)
class CryptoNNClient(object):
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
class CryptoNNServer(object):
    def __init__(self, n_output, n_features, n_hidden=30,
                 l1=0.0, l2=0.0, epochs=500, eta=0.001,
                 alpha=0.0, decrease_const=0.0,
                 mini_batches=1, smc=None):
        self.n_output = n_output
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.w1, self.w2 = self._initialize_weights()
        self.l1 = l1
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.alpha = alpha
        self.decrease_const = decrease_const
        self.mini_batches = mini_batches
        self.smc = smc
        self.parties = dict()

    def register(self, party):
        self.parties[party.get_id()] = party.get_features_size()

    def _initialize_weights(self):
        w1 = np.random.uniform(-1.0, 1.0, size=self.n_hidden*(self.n_features + 1))
        w1 = w1.reshape(self.n_hidden, self.n_features + 1)
        w2 = np.random.uniform(-1.0, 1.0, size=self.n_output*(self.n_hidden + 1))
        w2 = w2.reshape(self.n_output, self.n_hidden + 1)
        return w1, w2

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

    def _feedforward(self, X, w1, w2):
        z2 = w1.dot(X.T)
        a2 = self._sigmoid(z2)
        a2 = self._add_bias_unit(a2, how='row')
        z3 = w2.dot(a2)
        a3 = self._sigmoid(z3)
        return z2, a2, z3, a3

    def _feedforward_secure(self, ct_batch, w1, w2):
        # simulate the smc
        if isinstance(self.smc, Secure2PCServer):
            sk_w1 = self.smc.request_key_ndarray(w1)
            z2 = self.smc.execute_ndarray(sk_w1, ct_batch.tolist(), w1)
        elif isinstance(self.smc, EnhancedSecure2PCServer):
            sk_w1 = self.smc.request_key_ndarray(w1, {'type': 'mife', 'parties': self.parties})
            z2 = self.smc.execute_ndarray(sk_w1, ct_batch, w1, {'type': 'mife'})
        # end of smc
        a2 = self._sigmoid(z2)
        a2 = self._add_bias_unit(a2, how='row')
        z3 = w2.dot(a2)
        a3 = self._sigmoid(z3)
        return z2, a2, z3, a3

    def _L2_reg(self, lambda_, w1, w2):
        return (lambda_/2.0) * (np.sum(w1[:, 1:] ** 2) + np.sum(w2[:, 1:] ** 2))

    def _L1_reg(self, lambda_, w1, w2):
        return (lambda_/2.0) * (np.abs(w1[:, 1:]).sum() + np.abs(w2[:, 1:]).sum())

    def _get_cost(self, y_encode, output, w1, w2):
        term1 = - y_encode * (np.log(output))
        term2 = (1.0 - y_encode) * np.log(1.0 - output)
        cost = np.sum(term1 - term2)
        L1_term = self._L1_reg(self.l1, w1, w2)
        L2_term = self._L2_reg(self.l2, w1, w2)
        cost = cost + L1_term + L2_term
        return cost

    def compute_cost(self, AL, Y):
        m = Y.shape[1]
        # Compute loss from aL and y.
        cost = (1. / m) * (-np.dot(Y, np.log(AL).T) - np.dot(1 - Y, np.log(1 - AL).T))
        cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
        assert (cost.shape == ())
        return cost

    def _get_gradient(self, a1, a2, a3, z2, y_encode, w1, w2):
        # back-propagation
        sigma3 = a3 - y_encode
        z2 = self._add_bias_unit(z2, how='row')
        sigma2 = w2.T.dot(sigma3) * self._sigmoid_gradient(z2)
        sigma2 = sigma2[1:, :]
        grad1 = sigma2.dot(a1)
        grad2 = sigma3.dot(a2.T)

        # regularize
        grad1[:, 1:] += self.l2 * w1[:, 1:]
        grad1[:, 1:] += self.l1 * np.sign(w1[:, 1:])
        grad2[:, 1:] += self.l2 * w2[:, 1:]
        grad2[:, 1:] += self.l1 * np.sign(w2[:, 1:])

        return grad1, grad2

    def _get_gradient_secure(self, ct_batch, a2, a3, z2, y_encode, w1, w2):
        sigma3 = a3 - y_encode
        z2 = self._add_bias_unit(z2, how='row')
        sigma2 = w2.T.dot(sigma3) * self._sigmoid_gradient(z2)
        sigma2 = sigma2[1:, :]
        # using smc
        # grad1 = sigma2.dot(a1)
        if isinstance(self.smc, Secure2PCServer):
            sk_sigma2 = self.smc.request_key_ndarray(sigma2)
            grad1 = self.smc.execute_ndarray(sk_sigma2, ct_batch.tolist(), sigma2)
        elif isinstance(self.smc, EnhancedSecure2PCServer):
            sk_sigma2 = self.smc.request_key_ndarray(sigma2, {'type': 'sife'})
            grad1 = self.smc.execute_ndarray(sk_sigma2, ct_batch, sigma2, {'type': 'sife'})
        # end of smc
        grad2 = sigma3.dot(a2.T)

        # regularize
        grad1[:, 1:] += self.l2 * w1[:, 1:]
        grad1[:, 1:] += self.l1 * np.sign(w1[:, 1:])
        grad2[:, 1:] += self.l2 * w2[:, 1:]
        grad2[:, 1:] += self.l1 * np.sign(w2[:, 1:])

        return grad1, grad2

    def predict(self, X):
        if len(X.shape) != 2:
            raise AttributeError('X must be a [n_samples, n_features] array.\n'
                                 'Use X[:,None] for 1-feature classification,'
                                 '\nor X[[i]] for 1-sample classification')
        X = self._add_bias_unit(X, how='column')
        z2, a2, z3, a3 = self._feedforward(X, self.w1, self.w2)
        y_pred = np.argmax(z3, axis=0)
        return y_pred

    def fit(self, x, y, print_progress=False):
        train_loss_hist = list()

        delta_w1_prev = np.zeros(self.w1.shape)
        delta_w2_prev = np.zeros(self.w2.shape)

        for epoch in range(self.epochs):
            logger.info('Epoch: %d/%d start ... ' % (epoch + 1, self.epochs))
            if print_progress:
                print('Epoch: %d/%d' % (epoch+1, self.epochs))
            # adaptive learning rate
            self.eta /= (1 + self.decrease_const*epoch)

            for i in range(len(y)):
                # feedforward
                if self.smc:
                    if isinstance(self.smc, Secure2PCServer):
                        z2, a2, z3, a3 = self._feedforward_secure(x[0][i], self.w1, self.w2)
                        cost = self._get_cost(y_encode=y[i], output=a3, w1=self.w1, w2=self.w2)
                        train_loss_hist.append(cost)
                        grad1, grad2 = self._get_gradient_secure(ct_batch=x[1][i], a2=a2, a3=a3, z2=z2,
                                                                 y_encode=y[i], w1=self.w1, w2=self.w2)
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

                        z2, a2, z3, a3 = self._feedforward_secure(ct_ff_batch_lst, self.w1, self.w2)
                        cost = self._get_cost(y_encode=y[i], output=a3, w1=self.w1, w2=self.w2)
                        train_loss_hist.append(cost)
                        grad1, grad2 = self._get_gradient_secure(ct_batch=ct_bp_batch_lst, a2=a2, a3=a3, z2=z2,
                                                                 y_encode=y[i], w1=self.w1, w2=self.w2)
                else:
                    z2, a2, z3, a3 = self._feedforward(x[i], self.w1, self.w2)
                    cost = self._get_cost(y_encode=y[i], output=a3, w1=self.w1, w2=self.w2)
                    train_loss_hist.append(cost)
                    grad1, grad2 = self._get_gradient(a1=x[i], a2=a2, a3=a3, z2=z2,
                                                      y_encode=y[i], w1=self.w1, w2=self.w2)

                delta_w1, delta_w2 = self.eta * grad1, self.eta * grad2
                self.w1 -= (delta_w1 + (self.alpha * delta_w1_prev))
                self.w2 -= (delta_w2 + (self.alpha * delta_w2_prev))
                delta_w1_prev, delta_w2_prev = delta_w1, delta_w2

            logger.info('Epoch: %d/%d Done' % (epoch + 1, self.epochs))
        return train_loss_hist
