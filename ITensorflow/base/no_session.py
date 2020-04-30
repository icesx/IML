# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com

# tf2 中使用 @tf.function 不需要session
import tensorflow as tf

W = tf.Variable(tf.ones(shape=(2, 2)), name="W")
b = tf.Variable(tf.zeros(shape=(2)), name="b")


@tf.function
def forward(x):
    return W * x + b


out_a = forward([1, 0])
print(out_a)
