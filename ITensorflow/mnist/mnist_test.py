# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com

import tensorflow as tf
import tensorflow.keras.datasets.mnist

if __name__ == '__main__':
    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))
    saver = tf.train.Saver({'W': W,'b': b})
    mnist = mnist.read_data_sets("MNIST_data/",one_hot=True)
    with tf.Session() as sess:
        saver.restore(sess,'./model/mnist/mnist.ckpt')
        print(sess)

    # print sess.run(accuracy,feed_dict={x: mnist.test.images,y_: mnist.test.labels})
