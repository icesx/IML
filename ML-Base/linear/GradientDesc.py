# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com
import numpy as np


class GradientDesc:
    def __init__(self, all_x,
                 all_y,
                 h,
                 cost,
                 theta,
                 alpha=.01,
                 iterations=10,
                 step_callback={}):
        self.__all_x = all_x
        self.__all_y = all_y
        self.__h = h
        self.__cost = cost
        self.__theta = theta
        self.__alpha = alpha
        self.__iterations = iterations
        self.__step_callback = step_callback

    def gradient_desc(self):
        J = []
        for numbers in range(self.__iterations):
            m = self.__all_y.size
            for i in range(m):
                one_x = np.array([self.__all_x[i]])
                one_y = np.array([self.__all_y[i]])
                # one_x.dot(theta).flatten() 为计算获得的one_y‘
                #
                self.__theta[0] = self.__theta[0] - self.__alpha * (1 / m) * sum(
                    (self.__h(one_x, self.__theta) - one_y) * one_x)
                # must compare cost with befor costed.
                J.append(self.__cost(one_x, one_y, self.__theta))
            print('Cost: ' + str(J))
            print('theta:', self.__theta)
            self.__step_callback(self.__all_x, self.__all_y, theta=self.__theta, numbers=numbers)
