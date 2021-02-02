# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com
# 2/2/21
import os

import matplotlib.pyplot as plt
from tensorflow import keras

from tools.load_data import keras_load_data

class_names = ['T-shirt_top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle_boot']


def save_fashion_mnist(workdir):
    train_images, test_images, train_labels, test_labels = keras_load_data(keras.datasets.fashion_mnist)
    print(train_labels)
    index = 0
    for image, lable in zip(train_images, train_labels):
        workdir_folder = workdir + "/" + class_names[lable]
        if os.path.exists(workdir_folder) is False:
            os.mkdir(workdir_folder)
        plt.imsave(workdir_folder + "/" + str(index) + ".jpg", image)
        index += 1


if __name__ == '__main__':
    save_fashion_mnist("/WORK/datasset/fashion_mnist")
