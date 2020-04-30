# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com

import tensorflow as tf
import input_data

if __name__ == '__main__':
    x = tf.placeholder("float",[None,784])
    W = tf.Variable(tf.zeros([784,10]),name='W')
    b = tf.Variable(tf.zeros([10]),name='b')
    y = tf.nn.softmax(tf.matmul(x,W) + b)
    y_ = tf.placeholder("float",[None,10])
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    mnist_data = input_data.read_data_sets("MNIST_data/",one_hot=True)
    print(mnist_data)
    for i in range(1000):
        batch_xs,batch_ys = mnist_data.train.next_batch(100)
        sess.run(train_step,feed_dict={x: batch_xs,y_: batch_ys})
    print(tf.argmax(y,1))
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
    print ("accuracy",accuracy)
    saver = tf.train.Saver()
    saver.save(sess,'./model/mnist/mnist.ckpt')
