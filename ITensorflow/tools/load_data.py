# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com


def keras_load_data(dataset):
    """

    :type dataset: tensorflow.keras
    """
    (train_images, train_labels), (test_images, test_labels) = dataset.load_data()
    print(train_images.shape)
    print(train_labels)
    print(len(test_labels))
    # 归一化处理
    train_images, test_images = train_images / 255.0, test_images / 255.0
    return train_images, test_images, train_labels, test_labels
