# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def show_data():
    iris = pd.read_csv('ex1data2.txt')
    x = iris.iloc[:, 0].values
    y = iris.iloc[:, 1].values
    plt.figure()
    p2 = plt.scatter(x, y, marker='+', color='c', label='2', s=50)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("linear")
    plt.show()


def show_line(theta):
    x = np.arange(1, 50)
    y = theta * x
    plt.title("show line")
    plt.xlabel("x axis caption")
    plt.ylabel("y axis caption")
    plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    show_data()
