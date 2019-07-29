import sys
import numpy as np
import time
import csv

sys.path.append("..")
from nn.utils_secure_computation import SecureComputation
from nn.utils_smc import secure_inner_product_realn_multiprocessing_pool_cost, secure_inner_product_cost

def test_secure_computation_randint():
    print("test without multiporcessing")
    X = np.random.randint(1, 10, (784, 2))
    W = np.random.randint(1, 100, (4, 784))
    # print(X)
    # print(W)
    print('without sc:')
    print(W.dot(X))
    sc = SecureComputation(X.shape[0])
    start = time.clock()
    dot_result = sc.secure_inner_product(X, W)
    end = time.clock()
    print('with sc')
    print(dot_result)
    print('with sc, cost time (s): ' + str(end - start))

def test_secure_computation_randint_multipprocessing():
    print("test with multiporcessing")
    X = np.random.randint(1, 100, (784, 2))
    W = np.random.randint(1, 100, (2, 784))
    # print(X)
    # print(W)
    print('without sc:')
    print(W.dot(X))
    sc = SecureComputation(X.shape[0])
    start = time.time()
    dot_result = sc.secure_inner_product_multiprocessing(X, W)
    end = time.time()
    print('with sc')
    print(dot_result)
    print('with sc, cost time (s): ' + str(end - start))


def test_secure_computation_inner_product_compare():
    X = np.random.randint(1, 100, (200, 20))
    W = np.random.randint(1, 100, (2, 200))
    dot_result, enc_cost, key_cost, dec_cost = secure_inner_product_cost(X, W)
    print("%f, %f, %f " % (enc_cost, key_cost, dec_cost))
    # print(X)
    # print(W)
    print(W.dot(X))
    dot_result, enc_cost, key_cost, dec_cost = secure_inner_product_realn_multiprocessing_pool_cost(X, W)
    print(dot_result)
    print("%f, %f, %f " % (enc_cost, key_cost, dec_cost))


def test_secure_computation_inner_product_statistic():
    # test_max_vector_value_list = [10, 100]  # precision
    # test_max_vector_length_list = [200, 400, 600, 800, 1000] # #features
    # test_max_res_shape0_list = [10, 30, 50] # #neural of first hidden layer
    # test_max_res_shape1_list = [200, 400, 600, 800, 1000]  # samples
    test_max_vector_value_list = [10, 100]  # precision
    test_max_vector_length_list = [200, 800]  # #features
    test_max_res_shape0_list = [10]  # #neural of first hidden layer
    test_max_res_shape1_list = [200, 600, 1000]  # samples
    print("testing the scip in original program")
    with open('sc_ip_original_cost.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        for i in test_max_vector_length_list:
            sc = SecureComputation(i)
            for j in test_max_vector_value_list:
                # already tested
                # if i == 200 and j == 10:
                #     continue
                for m in test_max_res_shape0_list:
                    for n in test_max_res_shape1_list:
                        print("%d, %d, %d, %d" % (i, j, m, n))
                        X = np.random.randint(1, j, (i, n))
                        W = np.random.randint(1, j, (m, i))
                        # print(X)
                        # print(W)
                        # print("max_value[%d], max_length[%d], max_shape_res[%d, %d]" % (j, i, m, n))
                        dot_result, enc_cost, key_cost, dec_cost = sc.secure_inner_product_cost(X, W)
                        # print(dot_result)
                        print("%d, %d, %d, %d, %f, %f, %f" % (i, j, m, n, enc_cost, key_cost, dec_cost))
                        writer.writerow([i, j, m, n, enc_cost, key_cost, dec_cost])


def test_secure_computation_inner_product_multi_statistic():
    # test_max_vector_value_list = [10, 100]  # precision
    # test_max_vector_length_list = [200, 400, 600, 800, 1000]  # #features
    # test_max_res_shape0_list = [10, 30, 50]  # #neural of first hidden layer
    # test_max_res_shape1_list = [200, 400, 600, 800, 1000]  # samples
    test_max_vector_value_list = [10, 100]  # precision
    test_max_vector_length_list = [200, 800]  # #features
    test_max_res_shape0_list = [10]  # #neural of first hidden layer
    test_max_res_shape1_list = [200, 600, 1000]  # samples
    print("testing the scip in multiprocessing program")
    with open('sc_ip_multiprocess_cost.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        for i in test_max_vector_length_list:
            sc = SecureComputation(i)
            for j in test_max_vector_value_list:
                for m in test_max_res_shape0_list:
                    for n in test_max_res_shape1_list:
                        X = np.random.randint(1, j, (i, n))
                        W = np.random.randint(1, j, (m, i))
                        # print(X)
                        # print(W)
                        # print("max_value[%d], max_length[%d], max_shape_res[%d, %d]" % (j, i, m, n))
                        dot_result, enc_cost, key_cost, dec_cost = secure_inner_product_realn_multiprocessing_pool_cost(X, W)
                        # print(dot_result)
                        print("%d, %d, %d, %d, %f, %f, %f" % (i, j, m, n, enc_cost, key_cost, dec_cost))
                        writer.writerow([i, j, m, n, enc_cost, key_cost, dec_cost])


def test_secure_computation_inner_product_multi_statistic_v2():
    # test_max_vector_value_list = [10, 100]  # precision
    # test_max_vector_length_list = [200, 400, 600, 800, 1000]  # #features
    # test_max_res_shape0_list = [10, 30, 50]  # #neural of first hidden layer
    # test_max_res_shape1_list = [200, 400, 600, 800, 1000]  # samples
    test_max_vector_value_list = [10]  # precision 100
    test_max_vector_length_list = [200]  # #features
    test_max_res_shape0_list = [10]  # #neural of first hidden layer
    test_max_res_shape1_list = [200]  # samples
    print("testing the scip in multiprocessing program")
    with open('sc_ip_multiprocess_cost.csv', 'a+') as csvfile:
        writer = csv.writer(csvfile)
        i = 200
        j = 10
        m = 10
        n = 200
        X = np.random.randint(1, 10, (200, 200))
        W = np.random.randint(1, 10, (10, 200))
        sc = SecureComputation(X.shape[0])
        dot_result, enc_cost, key_cost, dec_cost = sc.secure_inner_product_realn_multiprocessing_cost(X, W)
        print("%d, %d, %d, %d, %f, %f, %f" % (i, j, m, n, enc_cost, key_cost, dec_cost))
        writer.writerow([i, j, m, n, enc_cost, key_cost, dec_cost])


def test_secure_computation_randn():
    X = np.random.randn(10,2)
    W = np.random.randn(2,10)
    # print(W)
    # print(X)
    print("without sc")
    print(W.dot(X))
    sc = SecureComputation(X.shape[0])
    start = time.time()
    # as ceiling, should be approximate result
    dot_result = sc.secure_inner_product_realn(X, W, 100, 100)
    end = time.time()
    print('with sc')
    print(dot_result)
    print('with sc, cost time (s): ' + str(end - start))


def test_secure_computation_mul_case_simple():
    Y = np.asarray([1, 0, 0, 0, 0, 0, 0, 1, 0, 0]).reshape(5,2)
    A = np.random.uniform(0,1, (5, 2))
    print("without sc")
    print(Y*A)
    sc = SecureComputation(10)
    start = time.time()
    mul_res = sc.secure_multiplication(Y, A, 1000, 1000)
    end = time.time()
    print('with sc')
    print(mul_res)
    print('with sc, cost time:' + str(end - start))


def test_secure_computation_mul_case2():
    Y = np.random.uniform(0, 1, (10, 20))
    A = np.random.uniform(0, 1, (10, 20))
    print("without sc")
    print(Y*A)
    sc = SecureComputation(10)
    start = time.time()
    mul_res = sc.secure_multiplication(Y, A, 1000, 1000)
    end = time.time()
    print('with sc')
    print(mul_res)
    print('with sc, cost time:' + str(end - start))


def test_secure_computation_mul_case3():
    Y = np.random.uniform(0, 1, (10, 20))
    A = np.random.uniform(0, 1, (10, 20))
    print("without sc")
    print(Y*A)
    sc = SecureComputation(10)
    start = time.time()
    mul_res = sc.secure_multiplication_multiprocessing(Y, A, 1000, 1000)
    end = time.time()
    print('with sc')
    print(mul_res)
    print('with sc, cost time:' + str(end - start))


def test_secure_computation_multiplication_statistic():
    # test_max_vector_value_list = [100, 200]  # precision
    # test_max_res_shape0_list = [10, 20]  # #neural of first hidden layer
    # test_max_res_shape1_list = [20]  # samples
    test_max_vector_value_list = [10, 100, 1000]  # precision
    test_max_res_shape0_list = [10]  # #neural of first hidden layer
    test_max_res_shape1_list = [200, 400, 600, 800, 1000]  # samples
    # test_max_res_shape0_list = [10]  # #neural of first hidden layer
    # test_max_res_shape1_list = [200, 400, 600, 800, 1000]  # samples
    print("testing the scm in original program")
    with open('sc_m_original_cost.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        sc = SecureComputation()
        for j in test_max_vector_value_list:
            for m in test_max_res_shape0_list:
                for n in test_max_res_shape1_list:
                    Y = np.random.uniform(-j, j, (m, n))
                    A = np.random.uniform(-j, j, (m, n))
                    dot_result, enc_cost, key_cost, dec_cost = sc.secure_multiplication_cost(Y, A, 1, 1)
                    # print(dot_result)
                    print("%d, %d, %d, %f, %f, %f" % (j, m, n, enc_cost, key_cost, dec_cost))
                    writer.writerow([j, m, n, enc_cost, key_cost, dec_cost])


def test_secure_computation_multiplication_multiprocess_statistic():
    test_max_vector_value_list = [300]  # precision
    test_max_res_shape0_list = [10]  # #neural of first hidden layer
    test_max_res_shape1_list = [20]  # samples
    # test_max_vector_value_list = [100, 200]  # precision
    # test_max_res_shape0_list = [10, 20, 30, 40, 50]  # #neural of first hidden layer
    # test_max_res_shape1_list = [200, 400, 600, 800, 1000]  # samples
    print("testing the scm in original program")
    with open('sc_m_multiprocessing_cost.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        sc = SecureComputation()
        for j in test_max_vector_value_list:
            for m in test_max_res_shape0_list:
                for n in test_max_res_shape1_list:
                    Y = np.random.uniform(0, j, (m, n))
                    A = np.random.uniform(0, j, (m, n))
                    # sc.secure_multiplication_multiprocessing_cost(Y, A, 1, 1)
                    dot_result, enc_cost, key_cost, dec_cost = sc.secure_multiplication_multiprocessing_cost(Y, A, 1, 1)
                    # print(dot_result)
                    print("%d, %d, %d, %f, %f, %f" % (j, m, n, enc_cost, key_cost, dec_cost))
                    writer.writerow([j, m, n, enc_cost, key_cost, dec_cost])
                    # dot_result, cost = sc.secure_multiplication_multiprocessing_cost2(Y, A, 1, 1)
                    # print("%d, %d, %d, %f" % (j, m, n, cost))
                    # writer.writerow([j, m, n, cost])


def test_secure_computation_sub_case1():
    # X = np.random.uniform(0, 1, (10, 1000))
    X = np.zeros((10,1000))
    Y = np.random.uniform(0, 1, (10, 1000))
    print("without sc")
    print(Y - X)
    sc = SecureComputation()
    start = time.time()
    mul_res = sc.secure_subtraction(Y, X, 1000)
    end = time.time()
    print('with sc')
    print(mul_res)
    print('with sc, cost time:' + str(end - start))


if __name__ == "__main__":
    # test_secure_computation_randint()
    # test_secure_computation_randint_multipprocessing()
    # test_secure_computation_randn()
    # test_secure_computation_mul_case_simple()
    # test_secure_computation_mul_case3()
    # test_secure_computation_inner_product_compare()
    # test_secure_computation_sub_case1()
    # test_secure_computation_randint_multipprocessing()
    # test_secure_computation_inner_product_multi_statistic_v2()

    # the following test for statistical result
    # test_secure_computation_inner_product_statistic()
    # test_secure_computation_multiplication_statistic()
    test_secure_computation_inner_product_multi_statistic()
    # test_secure_computation_multiplication_multiprocess_statistic()