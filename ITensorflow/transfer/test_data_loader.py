# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com
from transfer.data_download import DataLoader
from transfer.import_global import *

if __name__ == '__main__':
    data_loader = DataLoader(IMG_SIZE, BATCH_SIZE)

    plt.figure(figsize=(10, 8))
    i = 0
    for img, label in data_loader.get_random_raw_images(20):
        plt.subplot(4, 5, i + 1)
        plt.imshow(img)
        plt.title("{} - {}".format(data_loader.get_label_name(label), img.shape))
        plt.xticks([])
        plt.yticks([])
        i += 1
    plt.tight_layout()
    plt.show()
