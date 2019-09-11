import logging

import numpy as np

logger = logging.getLogger(__name__)

class Secure2PCClient(object):
    def __init__(self, crypto, precision):
        self.crypto_tpa, self.crypto_client = crypto
        self.precision = precision

    def execute(self, data_array):
        data_list = (data_array * pow(10, self.precision)).astype(int).flatten().tolist()
        pk = self.crypto_tpa.generate_public_key(len(data_list))
        ct_data = self.crypto_client.encrypt(pk, data_list)
        return ct_data

    def execute_ndarray(self, data_ndarray):
        assert type(data_ndarray) == np.ndarray, 'input data should be in numpy array format'
        assert len(data_ndarray.shape) == 2, 'at present, only address 2d array'

        ct_list = [self.execute(data_ndarray[i, :]) for i in range(data_ndarray.shape[0])]
        return ct_list

class Secure2PCServer(object):

    def __init__(self, crypto, precision):
        self.crypto_tpa, self.crypto_client = crypto
        self.precision_client, self.precision_server = precision
        self.common_pk = self.crypto_tpa.generate_common_public_key()

    def request_key(self, data_array):
        data_list = (data_array * pow(10, self.precision_server)).astype(int).flatten().tolist()
        sk = self.crypto_tpa.generate_private_key(data_list)
        return sk

    def execute(self, sk, ct, data_array):
        data_list = (data_array * pow(10, self.precision_server)).astype(int).flatten().tolist()
        max_inner_prod = 100000000 # max_value * max_value * self.vec_len
        dec_prod = self.crypto_client.decrypt(self.common_pk, sk, data_list, ct, max_inner_prod)
        if dec_prod is None:
            logger.debug('find a bad case - decryption: ')
            assert False
        return float(dec_prod)/pow(10, self.precision_server)/pow(10, self.precision_client)

    def request_key_ndarray(self, data_ndarray):
        assert type(data_ndarray) == np.ndarray, 'input weight should be a numpy array'
        assert len(data_ndarray.shape) == 2, 'only address 2d array'

        sk_list = [self.request_key(data_ndarray[i, :]) for i in range(data_ndarray.shape[0])]
        return sk_list

    def execute_ndarray(self, sk_list, ct_list, data_ndarray):
        assert type(data_ndarray) == np.ndarray, 'input weight should be a numpy array'
        assert len(data_ndarray.shape) == 2, 'only address 2d array'
        assert len(sk_list) == data_ndarray.shape[0]

        res = np.zeros((data_ndarray.shape[0], len(ct_list)))
        for i in range(data_ndarray.shape[0]):
            for j in range(len(ct_list)):
                res[i][j] = self.execute(sk_list[i], ct_list[j], data_ndarray[i, :])
        return res


