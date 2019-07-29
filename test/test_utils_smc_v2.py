import sys
import numpy as np
import time
import csv

sys.path.append("..")
from nn.utils_smc import smc_inner_product, smc_inner_product_float, smc_inner_product_parallel_cost, \
    smc_inner_product_parallel, smc_inner_product_parallel_float, smc_inner_product_cost, \
    smc_fundamental_operation, smc_fundamental_operation_cost, smc_fundamental_operation_float, \
    smc_fundamental_operation_parallel, smc_fundamental_operation_parallel_cost, \
    smc_fundamental_operation_parallel_float


def test_smc_inner_product():
    print("-->test inner product without multiporcessing")
    X = np.random.randint(1, 10, (50, 2))
    W = np.random.randint(1, 10, (2, 50))
    # print(X)
    # print(W)
    print('without crypto:')
    print(W.dot(X))
    start = time.clock()
    dot_result = smc_inner_product(X, W)
    end = time.clock()
    print('with crypto:')
    print(dot_result)
    print('with crypto, cost time (s): ' + str(end - start))


def test_smc_inner_product_parallel():
    print("-->test inner product with parallel processing")
    X = np.random.randint(1, 10, (50, 2))
    W = np.random.randint(1, 10, (2, 50))
    # print(X)
    # print(W)
    print('without crypto:')
    print(W.dot(X))
    start = time.clock()
    dot_result = smc_inner_product_parallel(X, W)
    end = time.clock()
    print('with crypto:')
    print(dot_result)
    print('with crypto, cost time (s): ' + str(end - start))


def test_smc_inner_product_float():
    print("-->test inner product with float")
    X = np.random.randn(10, 2)
    W = np.random.randn(2, 10)
    # print(W)
    # print(X)
    print("without crypto")
    print(W.dot(X))
    start = time.clock()
    # as ceiling, should be approximate result
    dot_result = smc_inner_product_parallel_float(X, W, 100, 100)
    end = time.clock()
    print('with crypto in float')
    print(dot_result)
    print('with sc, cost time (s): ' + str(end - start))


def test_smc_inner_product_parallel_float():
    print("-->test inner product with float in parallel")
    X = np.random.randn(10, 2)
    W = np.random.randn(2, 10)
    # print(W)
    # print(X)
    print("without crypto")
    print(W.dot(X))
    start = time.clock()
    # as ceiling, should be approximate result
    dot_result = smc_inner_product_float(X, W, 100, 100)
    end = time.clock()
    print('with crypto in float in parallel')
    print(dot_result)
    print('with sc, cost time (s): ' + str(end - start))


def test_smc_inner_product_parallel_compare():
    print("-->test inner product comparing with/without parallel ")
    X = np.random.randint(1, 100, (200, 20))
    W = np.random.randint(1, 100, (2, 200))
    dot_result, enc_cost, key_cost, dec_cost = smc_inner_product_cost(X, W)
    print("without parallel cost: %f, %f, %f " % (enc_cost, key_cost, dec_cost))
    # print(X)
    # print(W)
    # print(W.dot(X))
    # print(W.dot(X) == dot_result)
    dot_result, enc_cost, key_cost, dec_cost = smc_inner_product_parallel_cost(X, W)
    # print(dot_result)
    # print(W.dot(X) == dot_result)
    print("with parallel cost: %f, %f, %f " % (enc_cost, key_cost, dec_cost))


def test_smc_fundamental_operation():
    print("-->test fundamental operations")
    X = (np.random.randn(2, 2) * 1000).astype(int)
    Y = (np.random.randn(2, 2) * 1000).astype(int)
    # print(X)
    # print(W)
    op_add = 'addition'
    op_sub = 'subtract'
    op_mul = 'multiplication'
    op_div = 'division'

    print('--> addition without crypto:')
    print(X + Y)
    start = time.clock()
    dot_result = smc_fundamental_operation(X, Y, op_add)
    end = time.clock()
    print('--> addition with crypto:')
    print(dot_result)
    print(dot_result == (X + Y))
    print('--> addition with crypto, cost time (s): ' + str(end - start))

    print('--> subtract without crypto:')
    print(X - Y)
    start = time.clock()
    dot_result = smc_fundamental_operation(X, Y, op_sub)
    end = time.clock()
    print('--> subtract with crypto:')
    print(dot_result)
    print(dot_result == (X - Y))
    print('--> subtract with crypto, cost time (s): ' + str(end - start))

    print('--> multiplication without crypto:')
    print(X * Y)
    start = time.clock()
    dot_result = smc_fundamental_operation(X, Y, op_mul)
    end = time.clock()
    print('--> multiplication with crypto:')
    print(dot_result)
    print(dot_result == (X * Y))
    print('--> multiplication with crypto, cost time (s): ' + str(end - start))


def test_smc_fundamental_operation_float():
    print("-->test fundamental operations in float, approximate")
    X = np.random.randn(2, 2)
    Y = np.random.randn(2, 2)
    # print(X)
    # print(W)
    op_add = 'addition'
    op_sub = 'subtract'
    op_mul = 'multiplication'
    op_div = 'division'

    print('--> addition without crypto:')
    print(X + Y)
    start = time.clock()
    dot_result = smc_fundamental_operation_float(X, Y, op_add)
    end = time.clock()
    print('--> addition with crypto:')
    print(dot_result)
    # print(dot_result == (X+Y))
    print('--> addition with crypto, cost time (s): ' + str(end - start))

    print('--> subtract without crypto:')
    print(X - Y)
    start = time.clock()
    dot_result = smc_fundamental_operation_float(X, Y, op_sub)
    end = time.clock()
    print('--> subtract with crypto:')
    print(dot_result)
    # print(dot_result == (X - Y))
    print('--> subtract with crypto, cost time (s): ' + str(end - start))

    print('--> multiplication without crypto:')
    print(X * Y)
    start = time.clock()
    dot_result = smc_fundamental_operation_float(X, Y, op_mul)
    end = time.clock()
    print('--> multiplication with crypto:')
    print(dot_result)
    # print(dot_result == (X * Y))
    print('--> multiplication with crypto, cost time (s): ' + str(end - start))


def test_smc_fundamental_operation_parallel():
    print("-->test fundamental operations in parallel")
    X = (np.random.randn(2, 2) * 1000).astype(int)
    Y = (np.random.randn(2, 2) * 1000).astype(int)
    # print(X)
    # print(W)
    op_add = 'addition'
    op_sub = 'subtract'
    op_mul = 'multiplication'
    op_div = 'division'

    print('--> addition without crypto:')
    print(X + Y)
    start = time.clock()
    dot_result = smc_fundamental_operation_parallel(X, Y, op_add)
    end = time.clock()
    print('--> addition with crypto:')
    print(dot_result)
    print('--> addition with crypto, cost time (s): ' + str(end - start))

    print('--> subtract without crypto:')
    print(X - Y)
    start = time.clock()
    dot_result = smc_fundamental_operation_parallel(X, Y, op_sub)
    end = time.clock()
    print('--> subtract with crypto:')
    print(dot_result)
    print('--> subtract with crypto, cost time (s): ' + str(end - start))

    print('--> multiplication without crypto:')
    print(X * Y)
    start = time.clock()
    dot_result = smc_fundamental_operation_parallel(X, Y, op_mul)
    end = time.clock()
    print('--> multiplication with crypto:')
    print(dot_result)
    print('--> multiplication with crypto, cost time (s): ' + str(end - start))


def test_smc_fundamental_operation_parallel_float():
    print("-->test fundamental operations in parallel in float, approximate")
    X = np.random.randn(2, 2)
    Y = np.random.randn(2, 2)
    # print(X)
    # print(W)
    op_add = 'addition'
    op_sub = 'subtract'
    op_mul = 'multiplication'
    op_div = 'division'

    print('--> addition without crypto:')
    print(X + Y)
    start = time.clock()
    dot_result = smc_fundamental_operation_parallel_float(X, Y, op_add)
    end = time.clock()
    print('--> addition with crypto:')
    print(dot_result)
    # print(dot_result == (X+Y))
    print('--> addition with crypto, cost time (s): ' + str(end - start))

    print('--> subtract without crypto:')
    print(X - Y)
    start = time.clock()
    dot_result = smc_fundamental_operation_parallel_float(X, Y, op_sub)
    end = time.clock()
    print('--> subtract with crypto:')
    print(dot_result)
    # print(dot_result == (X - Y))
    print('--> subtract with crypto, cost time (s): ' + str(end - start))

    print('--> multiplication without crypto:')
    print(X * Y)
    start = time.clock()
    dot_result = smc_fundamental_operation_parallel_float(X, Y, op_mul)
    end = time.clock()
    print('--> multiplication with crypto:')
    print(dot_result)
    # print(dot_result == (X * Y))
    print('--> multiplication with crypto, cost time (s): ' + str(end - start))


def test_smc_fundamental_operation_parallel_compare():
    print("-->test fundamental computation comparing with/without parallel ")
    X = (np.random.randn(2, 2) * 1000).astype(int)
    Y = (np.random.randn(2, 2) * 1000).astype(int)
    op_mul = 'multiplication'
    res, enc_cost, key_cost, dec_cost = smc_fundamental_operation_cost(X, Y, op_mul)
    print("without parallel cost: %f, %f, %f " % (enc_cost, key_cost, dec_cost))
    # print(X)
    # print(W)
    # print(W.dot(X))
    # print(W.dot(X) == dot_result)
    res, enc_cost, key_cost, dec_cost = smc_fundamental_operation_parallel_cost(X, Y, op_mul)
    # print(dot_result)
    # print(W.dot(X) == dot_result)
    print("with parallel cost: %f, %f, %f " % (enc_cost, key_cost, dec_cost))


def test_smc_fundamental_operation_addition():
    test_max_vector_value_list = [10, 100, 1000, 10000]  # precision
    test_max_res_shape0_list = [10]  # #neural of first hidden layer
    test_max_res_shape1_list = [200, 400, 600, 800, 1000]  # samples
    print("testing the sc add")
    op = 'addition'
    with open('sc_add_original_cost.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        for j in test_max_vector_value_list:
            for m in test_max_res_shape0_list:
                for n in test_max_res_shape1_list:
                    X = np.random.uniform(-j, j, (m, n))
                    Y = np.random.uniform(-j, j, (m, n))
                    # print(X)
                    # print(W)
                    res, enc_cost, key_cost, dec_cost = smc_fundamental_operation_cost(X, Y, op)
                    # print(dot_result)
                    print("%d, %d, %d, %f, %f, %f" % (j, m, n, enc_cost, key_cost, dec_cost))
                    writer.writerow([j, m, n, enc_cost, key_cost, dec_cost])


def test_smc_fundamental_operation_addition_parallel():
    test_max_vector_value_list = [10, 100, 1000, 10000]  # precision
    test_max_res_shape0_list = [10]  # #neural of first hidden layer
    test_max_res_shape1_list = [200, 400, 600, 800, 1000]  # samples
    print("testing the sc add in parallel")
    op = 'addition'
    with open('sc_add_multiprocessing_cost.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        for j in test_max_vector_value_list:
            for m in test_max_res_shape0_list:
                for n in test_max_res_shape1_list:
                    X = np.random.uniform(-j, j, (m, n))
                    Y = np.random.uniform(-j, j, (m, n))
                    # print(X)
                    # print(W)
                    res, enc_cost, key_cost, dec_cost = smc_fundamental_operation_parallel_cost(X, Y, op)
                    # print(dot_result)
                    print("%d, %d, %d, %f, %f, %f" % (j, m, n, enc_cost, key_cost, dec_cost))
                    writer.writerow([j, m, n, enc_cost, key_cost, dec_cost])


def test_smc_fundamental_operation_mul():
    test_max_vector_value_list = [10, 100, 1000, 10000]  # precision
    test_max_res_shape0_list = [10]  # #neural of first hidden layer
    test_max_res_shape1_list = [200, 400, 600, 800, 1000]  # samples
    print("testing the sc multiplication")
    op = 'multiplication'
    with open('sc_mul_original_cost.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        for j in test_max_vector_value_list:
            for m in test_max_res_shape0_list:
                for n in test_max_res_shape1_list:
                    X = np.random.uniform(-j, j, (m, n))
                    Y = np.random.uniform(-j, j, (m, n))
                    # print(X)
                    # print(W)
                    res, enc_cost, key_cost, dec_cost = smc_fundamental_operation_cost(X, Y, op)
                    # print(dot_result)
                    print("%d, %d, %d, %f, %f, %f" % (j, m, n, enc_cost, key_cost, dec_cost))
                    writer.writerow([j, m, n, enc_cost, key_cost, dec_cost])


def test_smc_fundamental_operation_mul_parallel():
    test_max_vector_value_list = [10, 100, 1000, 10000]  # precision
    test_max_res_shape0_list = [10]  # #neural of first hidden layer
    test_max_res_shape1_list = [200, 400, 600, 800, 1000]  # samples
    print("testing the sc multiplication in parallel")
    op = 'multiplication'
    with open('sc_mul_multiprocessing_cost.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        for j in test_max_vector_value_list:
            for m in test_max_res_shape0_list:
                for n in test_max_res_shape1_list:
                    X = np.random.uniform(-j, j, (m, n))
                    Y = np.random.uniform(-j, j, (m, n))
                    # print(X)
                    # print(W)
                    res, enc_cost, key_cost, dec_cost = smc_fundamental_operation_parallel_cost(X, Y, op)
                    # print(dot_result)
                    print("%d, %d, %d, %f, %f, %f" % (j, m, n, enc_cost, key_cost, dec_cost))
                    writer.writerow([j, m, n, enc_cost, key_cost, dec_cost])


def test_smc_inner_product_stat():
    test_max_vector_value_list = [10, 100, 200]  # precision
    test_max_vector_length_list = [10, 100]  # #features
    test_max_res_shape0_list = [10]  # #neural of first hidden layer
    test_max_res_shape1_list = [200, 400, 600, 800, 1000]  # samples
    print("testing the scip in original program")
    with open('sc_ip_original_cost.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        for i in test_max_vector_length_list:
            for j in test_max_vector_value_list:
                for m in test_max_res_shape0_list:
                    for n in test_max_res_shape1_list:
                        # print("%d, %d, %d, %d" % (i, j, m, n))
                        X = np.random.randint(1, j, (i, n))
                        W = np.random.randint(1, j, (m, i))
                        dot_result, enc_cost, key_cost, dec_cost = smc_inner_product_cost(X, W)
                        # print(dot_result)
                        print("%d, %d, %d, %d, %f, %f, %f" % (i, j, m, n, enc_cost, key_cost, dec_cost))
                        writer.writerow([i, j, m, n, enc_cost, key_cost, dec_cost])

def test_smc_inner_product_parallel_stat():
    test_max_vector_value_list = [10, 100, 200]  # precision
    test_max_vector_length_list = [10, 100]  # #features
    test_max_res_shape0_list = [10]  # #neural of first hidden layer
    test_max_res_shape1_list = [200, 400, 600, 800, 1000]  # samples
    print("testing the scip in parallel program")
    with open('sc_ip_multiprocessing_cost.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        for i in test_max_vector_length_list:
            for j in test_max_vector_value_list:
                for m in test_max_res_shape0_list:
                    for n in test_max_res_shape1_list:
                        # print("%d, %d, %d, %d" % (i, j, m, n))
                        X = np.random.randint(1, j, (i, n))
                        W = np.random.randint(1, j, (m, i))
                        dot_result, enc_cost, key_cost, dec_cost = smc_inner_product_parallel_cost(X, W)
                        # print(dot_result)
                        print("%d, %d, %d, %d, %f, %f, %f" % (i, j, m, n, enc_cost, key_cost, dec_cost))
                        writer.writerow([i, j, m, n, enc_cost, key_cost, dec_cost])


def test():
    pass


if __name__ == "__main__":
    # test_smc_inner_product()
    # test_smc_inner_product_parallel()
    # test_smc_inner_product_float()
    # test_smc_inner_product_parallel_float()
    # test_smc_inner_product_parallel_compare()

    # test_smc_fundamental_operation()
    # test_smc_fundamental_operation_float()
    # test_smc_fundamental_operation_parallel()
    # test_smc_fundamental_operation_parallel_float()
    # test_smc_fundamental_operation_parallel_compare()
    # test_smc_fundamental_operation_addition()
    # test_smc_fundamental_operation_addition_parallel()
    # test_smc_fundamental_operation_mul()
    # test_smc_fundamental_operation_mul_parallel()
    # test_smc_inner_product_stat()
    test_smc_inner_product_parallel_stat()
    test()
