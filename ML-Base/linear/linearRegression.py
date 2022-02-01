# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com
from __future__ import division

import matplotlib.pyplot as plt
import numpy as np

from linear.GradientDesc import GradientDesc


def cost(x, y, theta=np.zeros((2, 1))):
    m = len(x)
    J = 1 / (2 * m) * sum((x.dot(theta).flatten() - y) ** 2)
    return J


def scatter_plot(x, y, theta=0.1, numbers=0):
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(x, y, marker='x')
    line_x = np.arange(0, 25)
    line_y = theta[0] * line_x
    plt.plot(line_x, line_y)
    plt.savefig(str(numbers) + '.png')


if __name__ == '__main__':
    # read data into array
    data = np.genfromtxt('ex1data2.txt', delimiter=',')
    all_x = data[:, 0]
    all_y = data[:, 1]
    GradientDesc(
        all_x=all_x,
        all_y=all_y,
        h=lambda one_x, theta: one_x.dot(theta).flatten(),
        cost=cost,
        theta=np.zeros((1, 1)),
        alpha=.01,
        iterations=10,
        step_callback=scatter_plot
    ).gradient_desc()
