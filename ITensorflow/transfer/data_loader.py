# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com
import tensorflow_datasets as tfds
import tensorflow as tf

class DataLoader(object):
    def __init__(self, image_size, batch_size):
        self.image_size = image_size
        self.batch_size = batch_size

        # 80% train data, 10% validation data, 10% test data
        split_weights = (8, 1, 1)
        splits = tfds.Split.TRAIN.subsplit(weighted=split_weights)

        (self.train_data_raw, self.validation_data_raw, self.test_data_raw), self.metadata = tfds.load(
            'cats_vs_dogs', split=list(splits),
            with_info=True, as_supervised=True)

        # Get the number of train examples
        self.num_train_examples = self.metadata.splits['train'].num_examples * 80 / 100
        self.get_label_name = self.metadata.features['label'].int2str

        # Pre-process data
        self._prepare_data()
        self._prepare_batches()

    # Resize all images to image_size x image_size
    def _prepare_data(self):
        self.train_data = self.train_data_raw.map(self._resize_sample)
        self.validation_data = self.validation_data_raw.map(self._resize_sample)
        self.test_data = self.test_data_raw.map(self._resize_sample)

    # Resize one image to image_size x image_size
    def _resize_sample(self, image, label):
        image = tf.cast(image, tf.float32)
        image = (image / 127.5) - 1
        image = tf.image.resize(image, (self.image_size, self.image_size))
        return image, label

    def _prepare_batches(self):
        self.train_batches = self.train_data.shuffle(1000).batch(self.batch_size)
        self.validation_batches = self.validation_data.batch(self.batch_size)
        self.test_batches = self.test_data.batch(self.batch_size)

    # Get defined number of  not processed images
    def get_random_raw_images(self, num_of_images):
        random_train_raw_data = self.train_data_raw.shuffle(1000)
        return random_train_raw_data.take(num_of_images)


