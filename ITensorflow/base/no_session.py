# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com

# tf2 中使用 @tf.function 不需要session
import tensorflow as tf

W = tf.constant([1.0, 2.0], name="W")
b = tf.constant([3.0, 4.0], name="b")


@tf.function
def forward(x):
    return W * x + b


out_a = forward([1, 0])
print(out_a)
