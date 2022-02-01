# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com

from tensorflow import keras

# imdm dataset
imdb = keras.datasets.imdb
(train_data, train_lables), (test_data, test_lables) = imdb.load_data(num_words=10000)
print("Train entries: {},lables{}".format(len(train_data), len(train_lables)))
print(train_data[0])
print("train_data[0],train_data[1] {}".format(train_data[0], train_lables[1]))

# convert the integers back to words

word_index = imdb.get_word_index()
print(word_index[0])
word_index = {k: (v + 3) for k, v in word_index.items()}
print(word_index)
