import random
import math
import time
import numpy as np

from crypto.fe import FEIP
from crypto.fe_simple import FEInnerProduct
from crypto.fe_simple import FEMultiplication
from crypto.fe_simple import FEFundamentalOperation
from charm.toolbox.integergroup import IntegerGroupQ, integer, IntegerGroup


def test_fe():
    n = 2
    max_value = 10
    x = [random.randint(0, max_value) for i in range(n)]
    y = [random.randint(0, max_value) for i in range(n)]
    print("x: " + str(x))
    print("y: " + str(y))

    checkprod = sum(map(lambda i: x[i] * y[i], range(n)))
    print('<x, y>: ' + str(checkprod))

    # p = integer(148829018183496626261556856344710600327516732500226144177322012998064772051982752493460332138204351040296264880017943408846937646702376203733370973197019636813306480144595809796154634625021213611577190781215296823124523899584781302512549499802030946698512327294159881907114777803654670044046376468983244647367)
    # q = integer(74414509091748313130778428172355300163758366250113072088661006499032386025991376246730166069102175520148132440008971704423468823351188101866685486598509818406653240072297904898077317312510606805788595390607648411562261949792390651256274749901015473349256163647079940953557388901827335022023188234491622323683)
    feip = FEIP(IntegerGroup())
    (mpk, msk) = feip.setup(n, secparam=256)
    ct = feip.encrypt(mpk, x)
    sk = feip.keyder(msk, y)
    dec_prod = feip.decrypt(mpk, ct, sk, y, max_value, checkprod)
    print('dec <x,y>: ' + str(dec_prod))
    # assert dec_prod == checkprod

def test_fe_inner_production():
    n = 100
    max_value = 600
    x = [random.randint(0, max_value) for i in range(n)]
    y = [random.randint(0, max_value) for i in range(n)]
    # x = [2, 2, 1]
    # y = [4, 1, 1]
    print("x: " + str(x))
    print("y: " + str(y))

    checkprod = sum(map(lambda i: x[i] * y[i], range(n)))
    print('<x, y>: ' + str(checkprod))
    feip = FEInnerProduct(IntegerGroup())
    (mpk, msk) = feip.setup(n, secparam=256)
    ct = feip.encrypt(mpk, x)
    sk = feip.keyder(msk, y)
    start = time.clock()
    dec_prod = feip.decrypt(mpk, ct, sk, y, max_value*max_value*n)
    end = time.clock()
    print('dec <x,y>: ' + str(dec_prod))
    print('cost time: ' + str(end - start))
    # assert dec_prod == checkprod

def test_fe_inner_production_numpy():
    n = 2
    max_value = 260
    # x = [random.randint(0, max_value) for i in range(n)]
    # y = np.random.randn(n,1)
    # y = [random.randint(0, max_value) for i in range(n)]
    x = np.asarray([1.8776171, -0.10520356, 1.22003044, -0.45423577, - 0.03169698, -1.40708356, -0.15832396, 2.90928895,
         -0.57279319, -0.10316766]).reshape(1,10)
    y = np.asarray([1.58310006, 0.3357981, -0.80354389, 2.10408785, 1.41193912, 0.11949047, 0.62438521, -0.6479802, 0.10802234,
         -1.41967142]).reshape(10,1)
    print("x: " + str(x))
    print("y: " + str(y))

    # checkprod = sum(map(lambda i: x[i] * y[i], range(n)))
    checkprod = x.dot(y)
    print('<x, y>: ' + str(checkprod))

    feip = FEInnerProduct(IntegerGroup())
    xx = (x * 100).astype(int).tolist()[0]
    yy = (y.reshape(1,10) * 100).astype(int).tolist()[0]
    print(xx)
    print(yy)
    print(sum(map(lambda i: xx[i] * yy[i], range(y.shape[0]))))
    (mpk, msk) = feip.setup(y.shape[0], secparam=256)
    ct = feip.encrypt(mpk, xx)
    sk = feip.keyder(msk, yy)
    start = time.clock()
    dec_prod = feip.decrypt(mpk, ct, sk, yy, max_value*max_value*n)
    dec_prod = dec_prod / (100*100)
    end = time.clock()
    print('dec <x,y>: ' + str(dec_prod))
    print('cost time: ' + str(end - start))
    # assert dec_prod == checkprod

def test_fe_multiplication():
    max_value = 260
    x = random.randint(0, max_value)
    # y = random.randint(0, max_value)
    y = 0.223
    print("x: " + str(x))
    print("y: " + str(y))
    checkprod = x * y
    print('xy: ' + str(checkprod))


    feip = FEMultiplication(IntegerGroup())
    (mpk, msk) = feip.setup(secparam=256)
    ct = feip.encrypt(mpk, x)
    y = int(y * 1000)
    sk = feip.keyder(msk, ct['commitment'], y)
    start = time.clock()
    dec_prod = feip.decrypt(mpk, ct, sk, y, max_value*max_value)
    dec_prod = dec_prod/1000
    end = time.clock()
    print('dec xy : ' + str(dec_prod))
    print('cost time: ' + str(end - start))
    # assert dec_prod == checkprod


def solve_dlog_bsgs(g, h, dlog_max):
  """
  Attempts to solve for the discrete log x, where g^x = h, using the Baby-Step
  Giant-Step algorithm. Assumes that x is at most dlog_max.
  """

  alpha = int(math.ceil(math.sqrt(dlog_max))) + 1
  g_inv = g ** -1
  tb = {}
  for i in range(alpha + 1):
    tb[(g ** (i * alpha)).__str__()] = i
    for j in range(alpha + 1):
      s = (h * (g_inv ** j)).__str__()
      if s in tb:
        i = tb[s]
        return i * alpha + j
  return -1

def solve_dlog_naive(g, h, dlog_max):
  """
  Naively attempts to solve for the discrete log x, where g^x = h, via trial and
  error. Assumes that x is at most dlog_max.
  """

  for j in range(dlog_max):
    if g ** j == h:
      return j
  return -1

def test_solve_dlog_bsgs():
    # group = IntegerGroup()
    # group.paramgen(256)
    # g = group.random()
    g = integer(24669293845903977339977463278253424825843996726142221513854325967692687060962, 102353942786081744009575637683568630313682804522154123705120102033097749062247)
    print(g)
    h = g ** 115
    print(h)
    print(solve_dlog_naive(g, h, 120))

def test_list():
    n = 5
    s = []
    for i in range(n):
        s.append(random.randint(1, 30))
    print(s)
    sum = 0
    for i in range(len(s)):
        sum += s[i]
    print(sum)


def test_fefo():
    group = IntegerGroup()
    p = 90841625992899044736915068676923590086910503646037290972161689497324782922043
    q = 45420812996449522368457534338461795043455251823018645486080844748662391461021
    fefo = FEFundamentalOperation(group, p, q)
    mpk, msk = fefo.setup(secparam=256)

    max_value = 20
    x = random.randint(-max_value, max_value)
    y = random.randint(-max_value, max_value)
    # x = 20
    # y = 2
    print('x = ' + str(x))
    print('y = ' + str(y))

    cmt, ct = fefo.encrypt(mpk, x)

    t1 = time.clock()
    op = 'addition'
    sk = fefo.keyder(mpk, msk, cmt, op, y)
    dlog_max = abs(x + y) + 1
    res = fefo.decrypt(mpk, sk, ct, op, y, dlog_max)
    t2 = time.clock()
    print('x + y = ' + str(x + y))
    print('x + y = %d (dec)' % res)
    print('time cost: %f' % (t2-t1))

    op = 'subtract'
    sk = fefo.keyder(mpk, msk, cmt, op, y)
    dlog_max = abs(x - y) + 1
    res = fefo.decrypt(mpk, sk, ct, op, y, dlog_max)
    t3 = time.clock()
    print('x - y = ' + str(x - y))
    print('x - y = %d (dec)' % res)
    print('time cost: %f' % (t3 - t2))

    op = 'multiplication'
    sk = fefo.keyder(mpk, msk, cmt, op, y)
    dlog_max = abs(x * y) + 1
    res = fefo.decrypt(mpk, sk, ct, op, y, dlog_max)
    t4 = time.clock()
    print('x * y = ' + str(x * y))
    print('x * y = %d (dec)' % res)
    print('time cost: %f' % (t4 - t3))

    op = 'division'
    sk = fefo.keyder(mpk, msk, cmt, op, y)
    dlog_max = int(x / y) + 1
    res = fefo.decrypt(mpk, sk, ct, op, y, dlog_max)
    t5 = time.clock()
    print('x / y = ' + str(x / y))
    print('x / y = ' + str(res) + ' (dec)')
    print('time cost: %f' % (t5 - t4))


def test_fef_serialize():
    group = IntegerGroup()
    p = 90841625992899044736915068676923590086910503646037290972161689497324782922043
    q = 45420812996449522368457534338461795043455251823018645486080844748662391461021
    fefo = FEFundamentalOperation(group, p, q)
    mpk, msk = fefo.setup(secparam=256)

    max_value = 20
    x = random.randint(-max_value, max_value)
    y = random.randint(-max_value, max_value)
    # x = 20
    # y = 2
    print('x = ' + str(x))
    print('y = ' + str(y))

    cmt, ct = fefo.encrypt_serialize(mpk, x)

    t1 = time.clock()
    op = 'addition'
    sk = fefo.keyder_serialize(mpk, msk, cmt, op, y)
    dlog_max = abs(x + y) + 1
    res = fefo.decrypt_serialize(mpk, sk, ct, op, y, dlog_max)
    t2 = time.clock()
    print('x + y = ' + str(x + y))
    print('x + y = %d (dec)' % res)
    print('time cost: %f' % (t2-t1))

    op = 'subtract'
    sk = fefo.keyder_serialize(mpk, msk, cmt, op, y)
    dlog_max = abs(x - y) + 1
    res = fefo.decrypt_serialize(mpk, sk, ct, op, y, dlog_max)
    t3 = time.clock()
    print('x - y = ' + str(x - y))
    print('x - y = %d (dec)' % res)
    print('time cost: %f' % (t3 - t2))

    op = 'multiplication'
    sk = fefo.keyder_serialize(mpk, msk, cmt, op, y)
    dlog_max = abs(x * y) + 1
    res = fefo.decrypt_serialize(mpk, sk, ct, op, y, dlog_max)
    t4 = time.clock()
    print('x * y = ' + str(x * y))
    print('x * y = %d (dec)' % res)
    print('time cost: %f' % (t4 - t3))

    op = 'division'
    sk = fefo.keyder_serialize(mpk, msk, cmt, op, y)
    dlog_max = int(x / y) + 1
    res = fefo.decrypt_serialize(mpk, sk, ct, op, y, dlog_max)
    t5 = time.clock()
    print('x / y = ' + str(x / y))
    print('x / y = ' + str(res) + ' (dec)')
    print('time cost: %f' % (t5 - t4))


# test_fe()
# test_solve_dlog_bsgs()
# test_list()
# test_fe_multiplication()
# test_fe_inner_production()
# test_fe_inner_production_numpy()
# test_fefo()
test_fef_serialize()