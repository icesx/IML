# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com
if __name__ == '__main__':
    from tools.save_module import load_module
    from tools.load_data import keras_load_data
    from tensorflow import keras
    import tensorflow as tf

    module = load_module("../tmp/save/keras_mnist")
    print(module)
    train_images, test_images, train_labels, test_labels = keras_load_data(keras.datasets.mnist)
    prediced = module.predict(test_images)
    print(test_labels[0], tf.argmax(prediced[0]))
