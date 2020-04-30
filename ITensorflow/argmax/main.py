# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com
import numpy as np
import tensorflow as tf

if __name__ == '__main__':
    pred = np.array([[31,23,4,24,27,34],
                     [18,3,25,0,6,35],
                     [28,14,33,22,20,8],
                     [13,30,21,19,7,9],
                     [16,1,26,32,2,29],
                     [17,12,5,11,10,15]])

    y = np.array([[31,23,4,24,27,34],
                  [18,3,25,0,6,35],
                  [28,14,3,22,20,8],
                  [13,30,21,19,7,9],
                  [16,1,26,32,2,29],
                  [17,12,5,11,10,15]])
    _pred = tf.argmax(pred,1)
    _y = tf.argmax(y,1)
    with tf.Session():
        print(_pred.eval())
        print(_y.eval())
        print(tf.equal(_pred,_y,name=None).eval())
