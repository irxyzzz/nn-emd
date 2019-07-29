import sys
sys.path.append("..")
from nn.utils_sc_multiplication import secure_multiplication_cost
import csv
import numpy as np


def test_secure_computation_multiplication_statistic():
    # test_max_vector_value_list = [300]  # precision
    # test_max_res_shape0_list = [10]  # #neural of first hidden layer
    # test_max_res_shape1_list = [20]  # samples
    #
    # test_max_vector_value_list = [100, 200]  # precision
    # test_max_res_shape0_list = [10, 20, 30, 40, 50]  # #neural of first hidden layer
    # test_max_res_shape1_list = [200, 400, 600, 800, 1000]  # samples

    test_max_vector_value_list = [10, 100, 1000]  # precision
    test_max_res_shape0_list = [10]  # #neural of first hidden layer
    test_max_res_shape1_list = [200, 400, 600, 800, 1000]  # samples

    print("testing the scm in original program")
    with open('sc_m_original_cost.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        for j in test_max_vector_value_list:
            for m in test_max_res_shape0_list:
                for n in test_max_res_shape1_list:
                    Y = np.random.uniform(0, j, (m, n))
                    A = np.random.uniform(0, j, (m, n))
                    # sc.secure_multiplication_multiprocessing_cost(Y, A, 1, 1)
                    dot_result, enc_cost, key_cost, dec_cost = secure_multiplication_cost(Y, A, 1, 1)
                    # print(dot_result)
                    print("%d, %d, %d, %f, %f, %f" % (j, m, n, enc_cost, key_cost, dec_cost))
                    writer.writerow([j, m, n, enc_cost, key_cost, dec_cost])
                    # dot_result, cost = sc.secure_multiplication_multiprocessing_cost2(Y, A, 1, 1)
                    # print("%d, %d, %d, %f" % (j, m, n, cost))
                    # writer.writerow([j, m, n, cost])


if __name__ == "__main__":
    test_secure_computation_multiplication_statistic()