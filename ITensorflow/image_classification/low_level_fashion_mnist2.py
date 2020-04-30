# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com

import tensorflow as tf
import matplotlib.pyplot as plt
if __name__ == '__main__':
    (train_image,train_label),(test_images,test_label) = tf.keras.datasets.fashion_mnist.load_data()
    #归一化处理
    train_image = train_image/255
    test_images = test_images/255
    #打印训练数据规模 结果为(60000, 28, 28),代表6w张图片
    print(train_image.shape)
    #引入模型
    model = tf.keras.Sequential()
    #把图像扁平成28*28的向量
    model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
    #隐藏层
    model.add(tf.keras.layers.Dense(128,activation='relu'))
    #输出层,输出10个概率值,使用softmax把十个输出变成一个概率分布
    model.add(tf.keras.layers.Dense(10,activation='softmax'))
    #编译模型,规定优化方法和损失函数,当标签使用的是数字编码，使用sparse_categorical_crossentropy这个损失函数
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['acc'])
    #训练模型，次数为5
    model.fit(train_image,train_label,epochs=10)
    #在测试数据上，对我们的模型进行评价
    print(model.evaluate(test_images,test_label))

