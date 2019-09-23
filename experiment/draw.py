import numpy as np
import matplotlib.pyplot as plt
import json


def test_draw_cost_time():
    # s2pc_cost_time = []
    # es2pc_cost_time = []
    # with open('logs/exp_nn_shallow_cs-2019-09-18-11-00.log', 'r') as infile:
    #     for line in infile:
    #         if 'training using secure2pc setting' in line:
    #             s2pc_cost_time.append(float(line.split(':')[2].split(' ')[1]))
    #         elif 'training using enhanced secure2pc setting' in line:
    #             es2pc_cost_time.append(float(line.split(':')[2].split(' ')[1]))

    es2pc_cost_time = [69.39, 69.19, 69.17, 68.84, 69.07, 68.7, 68.72, 68.42, 68.78, 69.03, 68.35, 68.04, 68.22, 68.62, 68.0, 75.86, 79.13, 74.49, 72.16, 71.57, 70.91, 70.14, 70.04, 69.45, 69.1, 68.96, 69.09, 68.71, 68.54, 68.24]
    s2pc_cost_time = [151.45, 121.43, 117.05, 115.34, 113.3, 112.38, 111.78, 112.05, 110.98, 111.47, 109.96, 109.7, 109.7, 109.79, 110.19, 108.97, 108.61, 121.68, 125.48, 117.83, 115.49, 113.58, 112.56, 111.82, 111.38, 109.94, 109.89, 110.28, 109.64, 109.32]
    s2pc_cost_time = s2pc_cost_time[4::5]
    es2pc_cost_time = es2pc_cost_time[4::5]
    x = [5, 10, 15, 20, 25, 30]

    plt.plot(x, np.array(s2pc_cost_time) / 5)
    plt.plot(x, np.array(es2pc_cost_time) / 5)
    plt.xlabel("# hidden layers")
    # plt.xticks(s2pc_cost_time, ['5', '10', '15', '20', '25', '30'])
    plt.ylabel("time (s)")
    # plt.title("training time of one mini-batch for different hidden layers")
    plt.legend(("NN-MEDS (HPT)",
                "NN-MEDS (VPT)"))
    plt.savefig('experiment/res/training_time_diff_layers.eps')

def test_draw_acc_diff_precision():
    acc_precision_lst = []
    with open('logs/acc_diff_precision.log', 'r') as infile:
        last_line_str = infile.readlines()[-1]
        acc_precision_lst = json.loads(last_line_str)
    x = [i for i in range(1, 51)]
    for y in acc_precision_lst:
        plt.plot(x, y)
    plt.xlabel("# iterations")
    # plt.xticks(s2pc_cost_time, ['5', '10', '15', '20', '25', '30'])
    plt.ylabel("accuracy")
    # plt.title("training time of one mini-batch for different hidden layers")
    plt.legend(("Precision 3",
                "Precision 5",
                "Precision 7",
                "Precision 9",))
    plt.savefig('experiment/res/acc_diff_precision.eps')


def test_draw_test_acc():
    test_acc_hist_base = None
    test_acc_hist_taylor = None
    test_acc_hist_fl_he = None
    test_acc_hist_fl_fe = None
    with open('../temp/8-9/__main___2019-08-09-14-54.log', 'r') as infile:
        for line in infile:
            if 'lr_gd_fe_test_acc_hist' in line:
                test_acc_hist_fl_fe = np.fromstring(next(infile)[1:-1], dtype=float, sep=',')
            elif 'lr_gd_appr_test_acc_hist' in line:
                test_acc_hist_taylor = np.fromstring(next(infile)[1:-1], dtype=float, sep=',')
            elif 'lr_gd_he_test_acc_hist' in line:
                test_acc_hist_fl_he = np.fromstring(next(infile)[1:-1], dtype=float, sep=',')
            elif 'lr_gd_test_acc_hist' in line:
                test_acc_hist_base = np.fromstring(next(infile)[1:-1], dtype=float, sep=',')

    plt.plot(test_acc_hist_base)
    plt.plot(test_acc_hist_taylor)
    plt.plot(test_acc_hist_fl_he)
    plt.plot(test_acc_hist_fl_fe)
    plt.xlabel("#iteration-batches")
    plt.ylabel("accuracy")
    plt.title("test accuracy")
    plt.legend(("Baseline - normal logistic regression",
                "Baseline - logistic regression in Taylor approx.",
                "Hardy et al.",
                "Our approach"))
    plt.savefig('../temp/8-9/cmp_test_acc_.eps')
    plt.close()