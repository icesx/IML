# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt


class Linear():
    def __init__(self):
        print "init linear..."

    def cost(self,x,y,theta=np.zeros((2,1))):
        m = len(x)
        J = 1 / (2 * m) * sum((x.dot(theta).flatten() - y) ** 2)
        return J

    def gradient_desc(self,all_x,all_y,h0,theta=np.zeros((1,1)),alpha=.01,iterations=10):
        J = []
        for numbers in range(iterations):
            m = all_y.size
            for i in range(m):
                one_x = np.array([all_x[i]])
                one_y = np.array([all_y[i]])
                # one_x.dot(theta).flatten() 为计算获得的one_y‘
                #
                theta[0] = theta[0] - alpha * (1 / m) * sum((h0(one_x,theta) - one_y) * one_x)
                # must compare cost with befor costed.
                J.append(self.cost(one_x,one_y,theta))
            print 'Cost: ' + str(J)
            print 'theta:',theta
            self.scatter_plot(all_x,all_y,theta=theta,numbers=numbers)
        return theta

    def scatter_plot(self,x,y,theta=0.1,numbers=0):
        plt.xlabel('x')
        plt.ylabel('y')
        plt.scatter(x,y,marker='x')
        line_x = np.arange(0,25)
        line_y = theta[0] * line_x
        plt.plot(line_x,line_y)
        plt.savefig(str(numbers) + '.png')


if __name__ == '__main__':
    # read data into array
    data = np.genfromtxt('ex1data2.txt',delimiter=',')
    all_x = data[:,0]
    all_y = data[:,1]
    theta = Linear().gradient_desc(all_x,all_y,lambda one_x,theta: one_x.dot(theta).flatten())
