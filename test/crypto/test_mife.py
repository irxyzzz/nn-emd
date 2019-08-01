import random
import time
import math
import os
import logging
import numpy as np
from contextlib import contextmanager

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.insert(0,parentdir)

from crypto.mife import MIFE

from keras.models import load_model

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

tpa_config_file = '../config/mife_p2_b8.json'
client_config_file = '../config/mife_p2_b8_dlog.json'

@contextmanager
def timer(ctx_msg):
    """Helper for measuring runtime"""
    time0 = time.perf_counter()
    yield
    logger.info('[%s][elapsed time: %.2f s]' % (ctx_msg, time.perf_counter() - time0))


def test_generate_config_file():
    logger.info('testing generating config files')
    func_value_bound = 100000000
    # func_value_bound = 100
    secparam = 256
    # 2 parties settings
    mife = MIFE(2)
    mife.generate_setup_config(tpa_config_file, secparam)
    mife.generate_dlog_table_config(client_config_file, func_value_bound)


def test_mife_basic():
    logger.info("testing the correctness of basic mife.")

    parties = 10
    vec_size = 1

    # prepare the test data
    max_test_value = 1000
    x = []
    for i in range(parties):
        x.append([random.randint(0, max_test_value) for m in range(vec_size)])
    y = [random.randint(0, max_test_value) for i in range(vec_size*parties)]
    logger.debug("x: %s" % str(x))
    logger.debug("y: %s" % str(y))
    check_prod = 0
    for i in range(parties):
        y_i = y[i*vec_size: (i+1)*vec_size]
        check_prod = check_prod + sum(map(lambda j: x[i][j] * y_i[j], range(vec_size)))
    logger.debug('original dot product <x,y>: %d' % check_prod)

    mife = MIFE(parties=parties, vec_size=vec_size)
    mife.setup(secparam=256)
    slot_pk_list = [mife.generate_public_key(i) for i in range(parties)]
    ct_list = [mife.encrypt(slot_pk_list[i], x[i]) for i in range(parties)]

    common_pk = mife.generate_public_key()
    sk = mife.generate_private_key(y)
    max_interprod = max_test_value * max_test_value * vec_size * parties
    with timer('total decryption time:') as t:
        dec_prod = mife.decrypt(common_pk, sk, y, ct_list, max_interprod)
        logger.debug('decrypted dot product <x,y>: %d' % dec_prod)


def test_mife_basic_with_config():
    # 1 passed in 764.24 seconds
    logger.info("testing the correctness of mife using config file.")

    parties = 2
    vec_size = 1

    # prepare the test data
    max_test_value = 1000
    x = []
    for i in range(parties):
        x.append([random.randint(0, max_test_value) for m in range(vec_size)])
    y = [random.randint(0, max_test_value) for i in range(vec_size*parties)]
    logger.debug("x: %s" % str(x))
    logger.debug("y: %s" % str(y))
    check_prod = 0
    for i in range(parties):
        y_i = y[i*vec_size: (i+1)*vec_size]
        check_prod = check_prod + sum(map(lambda j: x[i][j] * y_i[j], range(vec_size)))
    logger.debug('original dot product <x,y>: %d' % check_prod)

    mife = MIFE(tpa_config=tpa_config_file, client_config=client_config_file)
    # mife.setup(secparam=256)
    slot_pk_list = [mife.generate_public_key(i) for i in range(parties)]
    ct_list = [mife.encrypt(slot_pk_list[i], x[i]) for i in range(parties)]

    common_pk = mife.generate_public_key()
    sk = mife.generate_private_key(y)
    max_interprod = max_test_value * max_test_value * vec_size * parties
    with timer('total decryption time:') as t:
        dec_prod = mife.decrypt(common_pk, sk, y, ct_list, max_interprod)
        logger.debug('decrypted dot product <x,y>: %d' % dec_prod)


# def test_mife_batch():
#     print("test the correctness of mife decryption in batch setting.")
#
#     parties = 3
#     vec_size = 1
#
#     max_test_value = 100
#     nn_layer_size = 4
#     nn_layer_param_length = 50
#
#     x = list()
#     for i in range(parties):
#         weights = list()
#         for i in range(nn_layer_size):
#             weight_vec = []
#             for j in range(nn_layer_param_length):
#                 weight_vec.append(random.randint(1, max_test_value))
#             weights.append(weight_vec)
#         x.append(weights)
#     # y = [random.randint(1, max_test_value) for i in range(parties)]
#     y = [1 for i in range(parties)]
#     # print("x: ", x)
#     # print("y: ", y)
#
#     check_prod = 0
#     final_weights = list()
#     for i in range(nn_layer_size):
#         final_weights_vec = list()
#         for j in range(nn_layer_param_length):
#             # for k in range(len(x)):
#             #     final_weights_vec_element = final_weights_vec_element + x[k][i][j] * y[k]
#             final_weights_vec_element = sum(map(lambda k: x[k][i][j] * y[k], range(parties)))
#             final_weights_vec.append(final_weights_vec_element)
#         final_weights.append(final_weights_vec)
#     print('original <x,y>:')
#     print(np.array(final_weights)/parties)
#
#     ## generate the fe instance and setup
#     group = IntegerGroup()
#     p = 90841625992899044736915068676923590086910503646037290972161689497324782922043
#     q = 45420812996449522368457534338461795043455251823018645486080844748662391461021
#     mife = MIFE(group, p, q, parties, vec_size)
#     mpk, msk = mife.setup(secparam=256)
#
#     # encryption of each party
#     ct = dict()
#     ct['t'] = []
#     ct['c'] = []
#     enc_total = 0.0
#     for i in range(parties):
#         enc_start_time = time.process_time()
#         slot_mpk = mife.mpk_deliver(i)
#         ct_i = mife.batch_encrypt_party(slot_mpk, x[i])
#         enc_end_time = time.process_time()
#         enc_total = enc_total + (enc_end_time - enc_start_time)
#         ct['t'].append(ct_i['t'])
#         ct['c'].append(ct_i['c'])
#     print("encryption avg time: ", (enc_total/parties))
#
#     # decryption
#     sk = mife.key_generate(y)
#     start_time = time.process_time()
#     # start_time = time.perf_counter()
#     dec_prod = mife.batch_decrypt_ml(mpk, ct, sk, y, max_test_value*max_test_value*vec_size*parties)
#     end_time = time.process_time()
#     # end_time = time.perf_counter()
#     print("dec <x,y>: ")
#     print(dec_prod)
#     print("dec <x,y> cost time: %f s" %(end_time - start_time))
#
# def test_mife_batch_precision():
#     print("test the correctness of mife decryption in batch setting.")
#
#     parties = 3
#     vec_size = 1
#     precision = 4
#
#     nn_layer_size = 4
#     nn_layer_param_length = 50
#
#     x = list()
#     for i in range(parties):
#         weights = list()
#         for i in range(nn_layer_size):
#             weight_vec = []
#             for j in range(nn_layer_param_length):
#                 weight_vec.append(random.random())
#             weights.append(weight_vec)
#         x.append(weights)
#     # y = [random.randint(1, max_test_value) for i in range(parties)]
#     y = [1 for i in range(parties)]
#     # print("x: ", x)
#     # print("y: ", y)
#
#     check_prod = 0
#     final_weights = list()
#     for i in range(nn_layer_size):
#         final_weights_vec = list()
#         for j in range(nn_layer_param_length):
#             # for k in range(len(x)):
#             #     final_weights_vec_element = final_weights_vec_element + x[k][i][j] * y[k]
#             final_weights_vec_element = sum(map(lambda k: x[k][i][j] * y[k], range(parties)))
#             final_weights_vec.append(final_weights_vec_element)
#         final_weights.append(final_weights_vec)
#     print('original <x,y>:')
#     print(np.array(final_weights)/parties)
#
#     ## generate the fe instance and setup
#     group = IntegerGroup()
#     p = 90841625992899044736915068676923590086910503646037290972161689497324782922043
#     q = 45420812996449522368457534338461795043455251823018645486080844748662391461021
#     mife = MIFE(group, p, q, parties, vec_size)
#     mpk, msk = mife.setup(secparam=256)
#
#     # encryption of each party
#     ct = dict()
#     ct['t'] = []
#     ct['c'] = []
#     enc_total = 0.0
#     for i in range(parties):
#         enc_start_time = time.process_time()
#         slot_mpk = mife.mpk_deliver(i)
#         ct_i = mife.batch_encrypt_party_precision(slot_mpk, x[i], precision)
#         enc_end_time = time.process_time()
#         enc_total = enc_total + (enc_end_time - enc_start_time)
#         ct['t'].append(ct_i['t'])
#         ct['c'].append(ct_i['c'])
#     print("encryption avg time: ", (enc_total/parties))
#
#     # decryption
#     sk = mife.key_generate(y)
#     start_time = time.process_time()
#     # start_time = time.perf_counter()
#     dec_prod = mife.batch_decrypt_ml_precision(mpk, ct, sk, y, pow(10, precision)*1*vec_size*parties, precision)
#     end_time = time.process_time()
#     # end_time = time.perf_counter()
#     print("dec <x,y>: ")
#     print(dec_prod)
#     print("dec <x,y> cost time: %f s" %(end_time - start_time))
#
# def test_mife_batch_parallel():
#     print("test the correctness of mife decryption in batch parallel setting.")
#
#     parties = 10
#     vec_size = 1
#
#     max_test_value = 100
#     nn_layer_size = 5
#     nn_layer_param_length = 50
#
#     x = list()
#     for i in range(parties):
#         weights = list()
#         for i in range(nn_layer_size):
#             weight_vec = []
#             for j in range(nn_layer_param_length):
#                 weight_vec.append(random.randint(1, max_test_value))
#             weights.append(weight_vec)
#         x.append(weights)
#     # y = [random.randint(1, max_test_value) for i in range(parties)]
#     y = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
#     # print("x: ", x)
#     # print("y: ", y)
#
#     # TODO here, start
#     check_prod = 0
#     final_weights = list()
#     for i in range(nn_layer_size):
#         final_weights_vec = list()
#         for j in range(nn_layer_param_length):
#             # for k in range(len(x)):
#             #     final_weights_vec_element = final_weights_vec_element + x[k][i][j] * y[k]
#             final_weights_vec_element = sum(map(lambda k: x[k][i][j] * y[k], range(parties)))
#             final_weights_vec.append(final_weights_vec_element)
#         final_weights.append(final_weights_vec)
#     print('original <x,y>:', np.array(final_weights)/10)
#
#     ## generate the fe instance and setup
#     group = IntegerGroup()
#     p = 90841625992899044736915068676923590086910503646037290972161689497324782922043
#     q = 45420812996449522368457534338461795043455251823018645486080844748662391461021
#     mife = MIFE(group, p, q, parties, vec_size)
#     mpk, msk = mife.setup(secparam=256)
#
#     # encryption of each party
#     ct = dict()
#     ct['t'] = []
#     ct['c'] = []
#     enc_total = 0.0
#     for i in range(parties):
#         enc_start_time = time.process_time()
#         slot_mpk = mife.mpk_deliver(i, mpk, msk)
#         ct_i = mife.batch_encrypt_party(slot_mpk, x[i])
#         enc_end_time = time.process_time()
#         enc_total = enc_total + (enc_end_time - enc_start_time)
#         ct['t'].append(ct_i['t'])
#         ct['c'].append(ct_i['c'])
#     print("encryption avg time: ", (enc_total/parties))
#
#     # decryption
#     sk = mife.key_generate(msk, y)
#     start_time = time.process_time()
#     # start_time = time.perf_counter()
#     dec_prod = mife.batch_decrypt_ml_parallel_thread(mpk, ct, sk, y, max_test_value*max_test_value*vec_size*parties)
#     end_time = time.process_time()
#     # end_time = time.perf_counter()
#     print("dec <x,y>: ", dec_prod)
#     print("dec <x,y> cost time: %f s" %(end_time - start_time))
#
# def solve_dlog_bsgs(g, h, p):
#         m = math.ceil(math.sqrt(p-1))
#         hash_table = {pow(g, i, p): i for i in range(m)}
#         c = pow(g, m * (p-2), p)
#         for j in range(m):
#             y = (h * pow(c, j, p)) % p
#             if y in hash_table:
#                 return j * m + hash_table[y]
#         return None
#
# def test_correctness_solve_dlog_bsgs():
#     group = IntegerGroup()
#     group.p = 90841625992899044736915068676923590086910503646037290972161689497324782922043
#     group.q = 45420812996449522368457534338461795043455251823018645486080844748662391461021
#     group.r = 2
#     g = group.randomGen()
#     x = 4212272
#     h = g ** x
#     print('original x: ', x)
#     start_time = time.process_time()
#     xx = solve_dlog_bsgs(int(g), int(h), group.p)
#     end_time = time.process_time()
#     print('solved x: ', xx)
#     print('cost time: ', (end_time - start_time))
#
#
# def test_mife_model_store():
#
#     group = IntegerGroup()
#     p = 90841625992899044736915068676923590086910503646037290972161689497324782922043
#     q = 45420812996449522368457534338461795043455251823018645486080844748662391461021
#     parties = 10
#     vec_size = 1
#     func_value_bound = 100000000
#     mife = MIFE(group, p, q, parties, vec_size)
#     t1 = time.process_time()
#     mife.setup_store(256, "../config/mife_model_p10_b8.json", func_value_bound)
#     t2 = time.process_time()
#     print("setup cost time: %f s" %(t2 - t1))
#
# def test_generate_mife_models():
#
#     group = IntegerGroup()
#     p = 90841625992899044736915068676923590086910503646037290972161689497324782922043
#     q = 45420812996449522368457534338461795043455251823018645486080844748662391461021
#     parties = [6, 8, 10, 12, 14, 16, 18, 20]
#     vec_size = 1
#     func_value_bound = 100000000
#     for i in parties:
#         mife = MIFE(group, p, q, i, vec_size)
#         t1 = time.process_time()
#         filename = "../config/mife_model_p" + str(i) + "_b8.json"
#         mife.setup_store(256, filename, func_value_bound)
#         t2 = time.process_time()
#         print("setup cost time: %f s" %(t2 - t1))
#
# def test_mife_model_recover():
#     t1 = time.process_time()
#     mife = MIFE()
#     mife, mpk, mpk = mife.setup_recover('mife_model_p10_b8.json')
#     t2 = time.process_time()
#     print("setup cost time: %f s" %(t2 - t1))
#     print('g^-12345 case : ', mife.dlog_table[mife.group.serialize(mife.mpk['g']**(-12345)).decode()])
#     # print('g^0 in string: ', mife.group.serialize(mife.mpk['g']**1234567).decode())
#     # print(mife.group.deserialize('0:44:ip0uBBLtyYAmffQlVji6pEBaDW0hRPAPDndCrHmy9wI=:44:yNaJyraMEC4tXH3bRshYgN+SBGrKYm39ouYGEOSoqTs=:'.encode()))
#     # print(mife.mpk['g'])
#
# def test_mife_model_batch_precision():
#     print("test the correctness of mife model decryption in batch setting.")
#
#     parties = 5
#     vec_size = 1
#     precision = 7
#
#     nn_layer_size = 8
#     nn_layer_param_length = 1200
#
#     x = list()
#     for i in range(parties):
#         weights = list()
#         for i in range(nn_layer_size):
#             weight_vec = []
#             for j in range(nn_layer_param_length):
#                 weight_vec.append(random.random())
#             weights.append(weight_vec)
#         x.append(weights)
#     # y = [random.randint(1, max_test_value) for i in range(parties)]
#     y = [1 for i in range(parties)]
#     # print("x: ", x)
#     # print("y: ", y)
#
#     check_prod = 0
#     final_weights = list()
#     for i in range(nn_layer_size):
#         final_weights_vec = list()
#         for j in range(nn_layer_param_length):
#             # for k in range(len(x)):
#             #     final_weights_vec_element = final_weights_vec_element + x[k][i][j] * y[k]
#             final_weights_vec_element = sum(map(lambda k: x[k][i][j] * y[k], range(parties)))
#             final_weights_vec.append(final_weights_vec_element/parties)
#         final_weights.append(final_weights_vec)
#     print('original <x,y>[0][0]:', final_weights[0][0])
#     # print(np.array(final_weights)/parties)
#
#     ## generate the fe instance and setup
#     mife = MIFE()
#     t1_load = time.process_time()
#     mife, mpk, msk = mife.setup_recover('../config/mife_model_p5_b8.json')
#     t2_load = time.process_time()
#     print('load mife model from file successfully. cost time %f s' % (t2_load - t1_load))
#
#     # encryption of each party
#     ct = dict()
#     ct['t'] = []
#     ct['c'] = []
#     enc_total = 0.0
#     for i in range(parties):
#         enc_start_time = time.process_time()
#         slot_mpk = mife.mpk_deliver(i)
#         ct_i = mife.batch_encrypt_party_precision(slot_mpk, x[i], precision)
#         enc_end_time = time.process_time()
#         enc_total = enc_total + (enc_end_time - enc_start_time)
#         ct['t'].append(ct_i['t'])
#         ct['c'].append(ct_i['c'])
#     print("encryption avg time: ", (enc_total/parties))
#
#     # decryption
#     sk = mife.key_generate(y)
#     start_time = time.process_time()
#     # start_time = time.perf_counter()
#     dec_prod = mife.batch_decrypt_ml_precision(mpk, ct, sk, y, pow(10, precision)*1*vec_size*parties, precision)
#     end_time = time.process_time()
#     # end_time = time.perf_counter()
#     print("dec <x,y>[0][0]: ", dec_prod[0][0])
#     # print(dec_prod)
#     print("dec <x,y> cost time: %f s" %(end_time - start_time))
#
# def test_mife_recover_with_enc_dec():
#     t1 = time.process_time()
#     mife = MIFE()
#     mife, mpk, msk = mife.setup_recover('../config/mife_model_p10_b8.json')
#     t2 = time.process_time()
#     print("setup cost time: %f s" %(t2 - t1))
#
#     parties = 10
#     precision = 4
#
#     nn_layer_size = 2
#     nn_layer_param_length = 3
#
#     x = list()
#     for i in range(parties):
#         weights = list()
#         for i in range(nn_layer_size):
#             weight_vec = []
#             for j in range(nn_layer_param_length):
#                 weight_vec.append(random.random())
#             weights.append(weight_vec)
#         x.append(weights)
#     # y = [random.randint(1, max_test_value) for i in range(parties)]
#     y = [1 for i in range(parties)]
#
#     final_weights = list()
#     for i in range(nn_layer_size):
#         final_weights_vec = list()
#         for j in range(nn_layer_param_length):
#             final_weights_vec_element = sum(map(lambda k: x[k][i][j] * y[k], range(parties)))
#             final_weights_vec.append(final_weights_vec_element)
#         final_weights.append(final_weights_vec)
#     print('original <x,y>:')
#     print(np.array(final_weights)/parties)
#
#     # encryption of each party
#     fe_ct_weights = None
#     enc_total = 0.0
#     for party_index in range(parties):
#         enc_start_time = time.process_time()
#         slot_mpk = mife.mpk_deliver(party_index)
#         ct_i = mife.batch_encrypt_party_precision(slot_mpk, x[party_index], precision)
#         enc_end_time = time.process_time()
#         enc_total = enc_total + (enc_end_time - enc_start_time)
#
#         if fe_ct_weights is None:
#             # initialize the fe_ct_weights
#             fe_ct_weights = dict()
#             fe_ct_weights['t'] = [None for k in range(parties)]
#             fe_ct_weights['t'][party_index] = ct_i['t']
#
#             cipher_weights = ct_i['c']
#             collect_cipher_weights = list()
#             for i in range(len(cipher_weights)):
#                 cipher_weight_vec = cipher_weights[i]
#                 collect_cipher_weight_vec = np.array(cipher_weights[i], dtype='object')
#                 for j, weight in np.ndenumerate(cipher_weight_vec):
#                     collect_cipher_weight_vec[j] = [None for k in range(parties)]
#                     collect_cipher_weight_vec[j][party_index] = weight
#                 collect_cipher_weights.append(collect_cipher_weight_vec)
#             fe_ct_weights['c'] = collect_cipher_weights
#         else:
#             fe_ct_weights['t'][party_index] = ct_i['t']
#             cipher_weights = ct_i['c']
#             for i in range(len(cipher_weights)):
#                 cipher_weight_vec = cipher_weights[i]
#                 for j, weight in np.ndenumerate(cipher_weight_vec):
#                     fe_ct_weights['c'][i][j][party_index] = weight
#     print("encryption avg time: ", (enc_total/parties))
#
#     # decryption
#     sk = mife.key_generate(y)
#     start_time = time.process_time()
#     # start_time = time.perf_counter()
#     final_weights = mife.batch_decrypt_precision(mpk, fe_ct_weights, sk, y, pow(10, precision)*1*parties, precision)
#     end_time = time.process_time()
#     # end_time = time.perf_counter()
#     print("dec <x,y>: ")
#     print(final_weights)
#     print("dec <x,y> cost time: %f s" %(end_time - start_time))
#
# def test_for_debug():
#     mife = MIFE()
#     mife, mpk, msk = mife.setup_recover('../config/mife_model_p10_b8.json')
#     test_p = 90841625992899044736915068676923590086910503646037290972161689497324782922043
#     print(test_p == mife.group.p)
#     gg = integer(61620851492855597346606601754363094779063488144119242500501469515803985028451, mife.group.p)
#     hh = integer(85202263960955468502003895584699005346003441583034999446108148931011498222081, mife.group.p)
#     res = mife.solve_dlog_naive(gg, hh, 1000000)
#     print(res)
#
#
#
# def test_check_real_model_weights():
#     print("test the correctness of mife decryption in batch setting using real model weights.")
#     # model = Sequential()
#     model = load_model('../config/mnist_cnn_model.h5')
#     # print(model.get_weights()[0][0])
#     parties = 5
#     vec_size = 1
#
#     weights_lst = [model.get_weights().copy() for i in range(parties)]
#     y = [1 for i in range(parties)]
#
#     print('cnn model weights info:')
#     print("nn layers: ", len(weights_lst[0]))
#     nn_layer_size = len(weights_lst[0])
#     for i in range(nn_layer_size):
#         print("nn layer-"+str(i), 'shape is', weights_lst[0][i].shape)
#
#     # result
#     # nn layers:  8
#     # nn layer-0 shape is (3, 3, 1, 32)
#     # nn layer-1 shape is (32,)
#     # nn layer-2 shape is (3, 3, 32, 64)
#     # nn layer-3 shape is (64,)
#     # nn layer-4 shape is (9216, 128)
#     # nn layer-5 shape is (128,)
#     # nn layer-6 shape is (128, 10)
#     # nn layer-7 shape is (10,)
#
#     # reform the weight list
#     # weights_lst_reform = [None for i in range(nn_layer_size)]
#     # for idx in range(parties):
#     #     party_weights = weights_lst[idx]
#     #     for i in range(nn_layer_size):
#     #         weights_vec = party_weights[i]
#     #         if weights_lst_reform[i] is None:
#     #             weights_vec_reform = np.array(weights_vec, dtype='object')
#     #             for j, weight in np.ndenumerate(weights_vec):
#     #                 weights_vec_reform[j] = [None for k in range(parties)]
#     #                 weights_vec_reform[j][idx] = weight
#     #             weights_lst_reform[i] = weights_vec_reform
#     #         else:
#     #             weights_vec_reform = weights_lst_reform[i]
#     #             for j, weight in np.ndenumerate(weights_vec):
#     #                 weights_vec_reform[j][idx] = weight
#
#     # calculate the avg weight
#     # final_weights_lst = list()
#     # for i in range(len(weights_lst_reform)):
#     #     weights_vec = weights_lst_reform[i]
#     #     final_weights_vec = np.array(weights_vec, dtype='object')
#     #     for j, weight in np.ndenumerate(weights_vec):
#     #         final_weights_vec[j] = np.sum(weight) / parties
#     #     final_weights_lst.append(final_weights_vec)
#     # print('original weights:', model.get_weights())
#     # print('final weights:', final_weights_lst)


# if __name__ == '__main__':
    # test_mife()
    # test_compare_solve_dlog()
    # test_mife_batch()
    # test_mife_batch_parallel()
    
    # test_mife_batch_precision()

    # test_mife_model_store()
    # test_generate_mife_models()
    # test_mife_model_recover()
    # test_mife_recover_with_enc_dec()

    # test_mife_model_batch_precision()

    # test_for_debug()

    # test_check_real_model_weights()
