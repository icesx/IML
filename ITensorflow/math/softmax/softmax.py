# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com


def base(z):
    import math
    z = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0]
    z_exp = [math.exp(i) for i in z]
    print(z_exp)  # Result: [2.72, 7.39, 20.09, 54.6, 2.72, 7.39, 20.09]
    sum_z_exp = sum(z_exp)
    print("sum_exp:", (sum_z_exp))  # Result: 114.98
    softmax = [round(i / sum_z_exp, 3) for i in z_exp]
    print(softmax)  # Result: [0.024, 0.064, 0.175, 0.475, 0.024, 0.064, 0.175]
    return softmax


def by_numpy(z):
    """

    :type z: list
    """
    import numpy as np
    _z = np.array(z)
    return (np.exp(_z) / sum(np.exp(_z)))


if __name__ == '__main__':
    z = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0]
    print("base:", base(z))
    print("numpy:", by_numpy(z))
