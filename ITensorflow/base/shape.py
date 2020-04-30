# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com
import tensorflow as tf

a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([3.0, 4.0], name="b")
result = a + b
print(result)
print("shape=", result.shape)

result2 = tf.add(a, b, name="and2")
print("result2=", result2)
print("shape=", result2.shape)