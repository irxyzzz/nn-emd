
import random
import numpy as np
import time

import os
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.insert(0,parentdir)

from crypto.paillier import GeneralPaillier
from keras.models import load_model



debug = False

def test_paillier_debug():
    paillier = GeneralPaillier()
    pk, sk = paillier.keygen(256)
    print(pk)
    print(sk)

def test_paillier_basic():
    paillier = GeneralPaillier()
    pk, sk = paillier.keygen(256)
    msg_1 = 123.456
    msg_2 = -124.456
    print('original message 1: ', msg_1)
    print('original message 2: ', msg_2)
    c1 = paillier.encrypt_float(pk, msg_1)
    c2 = paillier.encrypt_float(pk, msg_2)
    ct_list = []
    ct_list.append(c1)
    ct_list.append(c2)
    # print(ct_list)
    ct_fusion = paillier.fuze(pk, ct_list)
    decipher = paillier.decrypt_float(pk, sk, ct_fusion)
    print('expcted result', (msg_2+msg_1))
    print('decrypted fuzed ciphertext', decipher)


def test_paillier_fuze_pt_ct_list():
    paillier = GeneralPaillier()
    pk, sk = paillier.keygen(256)
    msg_1 = [1, -1, 0]
    msg_2 = [4.810541759066682, 2.316219327696398, 4.032863915958192]

    ct_list = [paillier.ext_encrypt(pk, msg_1[i]) for i in range(3)]
    print(ct_list)
    ct_fusion = paillier.fuze_pt_ct_float_list(pk, ct_list, msg_2, precision=3)
    print(ct_fusion)
    decipher = paillier.ext_decrypt_float_list(pk, sk, ct_fusion)
    print('expcted result', (np.array(msg_1) * np.array(msg_2)))
    print('decrypted fuzed ciphertext', decipher)

def test_paillier_fuze_pt_ct_double():
    paillier = GeneralPaillier()
    pk, sk = paillier.keygen(256)
    msg_1 = 1
    msg_2 = 4.810
    msg_3 = -2.342

    ct_list = paillier.ext_encrypt(pk, msg_1)
    ct_fusion1 = paillier.fuze_pt_ct_float(pk, ct_list, msg_2, precision=3)
    ct_fusion2 = paillier.fuze_pt_ct_float(pk, ct_fusion1, msg_3, precision=3)
    decipher = paillier.ext_decrypt_float(pk, sk, ct_fusion2, precision=6)
    print('expcted result', (msg_1 * msg_2 * msg_3))
    print('decrypted fuzed ciphertext', decipher)


def test_paillier_fuze_pt_ct_addition():
    paillier = GeneralPaillier()
    pk, sk = paillier.keygen(256)
    msgs = [0, 2, -3, 4, 2]
    msgs = [0, 2, 3, 4, -2]
    msgs = [0, -2, -3, 4, 2]
    msgs = [0, -2, -3, 4, -2]
    msgs = [0, -2, -3, -4, 2]

    ct1 = paillier.ext_encrypt(pk, msgs[1])
    ct3 = paillier.ext_encrypt(pk, msgs[3])
    ct_fusion1 = paillier.fuze_pt_ct_float(pk, ct1, msgs[2], precision=3)
    ct_fusion2 = paillier.fuze_pt_ct_float(pk, ct3, msgs[4], precision=3)
    ct_fusion = paillier.ext_fuze_two_ct_list(pk, sk, [ct_fusion1], [ct_fusion2])
    decipher = paillier.ext_decrypt_float_list(pk, sk, ct_fusion, precision=3)
    print('expcted result', (msgs[1] * msgs[2] + msgs[3] * msgs[4]))
    print('decrypted fuzed ciphertext', decipher)


def test_paillier_fuze_mtx_ct():
    paillier = GeneralPaillier()
    pk, sk = paillier.keygen(256)
    mtx = np.array([1,-0.451,-1,1,0,-0.345])
    # mtx = np.array([1,-0.451,1,0,0,0])
    msg_2 = mtx.reshape(2,3)
    msg_1 = [1,1,3]

    ct_1_list = [paillier.ext_encrypt(pk, i) for i in msg_1]
    # print(ct_1_list)
    ct_fusion = paillier.fuze_matrix_ct_list(pk, msg_2, ct_1_list, precision=3)
    # print(ct_fusion)
    decipher = paillier.ext_decrypt_float_list(pk, sk, ct_fusion, precision=3)
    print(msg_2)
    print(np.array(msg_1))
    print('expcted result', (np.dot(msg_2, np.array(msg_1))))
    print('decrypted fuzed ciphertext', decipher)

def test_paillier_fuze_two_ct_list():
    paillier = GeneralPaillier()
    pk, sk = paillier.keygen(256)
    msg1 = [1,-0.451,1]
    msg2 = [1,2,3]
    msg3 = [1,0,-0.145]
    msg4 = [2,1,3]

    ct_2_list = [paillier.ext_encrypt(pk, i) for i in msg2]
    ct_fusion1 = paillier.fuze_pt_ct_float_list(pk, ct_2_list, msg1, precision=3)
    ct_4_list = [paillier.ext_encrypt(pk, i) for i in msg4]
    ct_fusion2 = paillier.fuze_pt_ct_float_list(pk, ct_4_list, msg3, precision=3)
    ct_fusion = paillier.ext_fuze_two_ct_list(pk, sk, ct_fusion1, ct_fusion2)

    decipher = paillier.ext_decrypt_float_list(pk, sk, ct_fusion, precision=3)

    print('expcted result', (np.array(msg1) * np.array(msg2) + np.array(msg3) * np.array(msg4)))
    print('decrypted fuzed ciphertext', decipher)

def test_paillier_ext_fuze_two_ct_list():
    paillier = GeneralPaillier()
    pk, sk = paillier.keygen(256)
    msg1 = [1, -0.451, 1]
    msg2 = [1, 2, 3]
    msg3 = [1, 0, -0.145]
    msg4 = [2, 1, 3]

    ct_2_list = [paillier.ext_encrypt(pk, i) for i in msg2]
    ct_fusion_list1 = paillier.fuze_pt_ct_float_list(pk, ct_2_list, msg1, precision=3)
    ct_4_list = [paillier.ext_encrypt(pk, i) for i in msg4]
    ct_fusion_list2 = paillier.fuze_pt_ct_float_list(pk, ct_4_list, msg3, precision=3)
    ct_fusion = paillier.ext_fuze_two_ct_list(pk, sk, ct_fusion_list1, ct_fusion_list2)

    print(paillier.ext_decrypt_float_list(pk, sk, ct_fusion, precision=3))

    mtx = np.array([1, -0.451, -1, 1, 0, -0.345]).reshape(2, 3)
    ct_mtx_fusion = paillier.fuze_matrix_ct_list(pk, mtx, ct_fusion, precision=3)

    decipher = paillier.ext_decrypt_float_list(pk, sk, ct_mtx_fusion, precision=6)

    exp1 = (np.array(msg1) * np.array(msg2) + np.array(msg3) * np.array(msg4))
    print(exp1)
    print(mtx)

    print('expcted result', np.dot(mtx, exp1.reshape(3,1)))
    print('decrypted fuzed ciphertext', decipher)


if __name__ == '__main__':

    # test_paillier_debug()
    # test_paillier_basic()
    # test_paillier_fuze_plaintext()
    # test_paillier_fuze_pt_ct_list()
    # test_paillier_fuze_pt_ct_double()
    # test_paillier_fuze_mtx_ct()
    test_paillier_fuze_two_ct_list()



