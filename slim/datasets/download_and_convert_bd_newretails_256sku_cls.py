# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Downloads and converts Flowers data to TFRecords of TF-Example protos.

This module downloads the Flowers data, uncompresses it, reads the files
that make up the Flowers data and creates two TFRecord datasets: one for train
and one for test. Each TFRecord dataset is comprised of a set of TF-Example
protocol buffers, each of which contain a single image and label.

The script should take about a minute to run.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys

import tensorflow as tf

from datasets import dataset_utils

# The URL where the Flowers data can be downloaded.
_DATA_URL = 'http://download.tensorflow.org/example_images/flower_photos.tgz'

# The number of images in the validation set.
_NUM_VALIDATION = 350

# Seed for repeatability.
_RANDOM_SEED = 0

# The number of shards per dataset split.
_NUM_SHARDS = 10


class ImageReader(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def read_image_dims(self, sess, image_data):
        image = self.decode_jpeg(sess, image_data)
        return image.shape[0], image.shape[1]

    def decode_jpeg(self, sess, image_data):
        image = sess.run(self._decode_jpeg,
                         feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def _get_filenames_and_classes(dataset_dir):
    """Returns a list of filenames and inferred class names.

    Args:
      dataset_dir: A directory containing a set of subdirectories representing
        class names. Each subdirectory should contain PNG or JPG encoded images.

    Returns:
      A list of image file paths, relative to `dataset_dir` and the list of
      subdirectories, representing class names.
    """
    flower_root = os.path.join(dataset_dir, 'flower_photos')
    directories = []
    class_names = []
    for filename in os.listdir(flower_root):
        path = os.path.join(flower_root, filename)
        if os.path.isdir(path):
            directories.append(path)
            class_names.append(filename)

    photo_filenames = []
    for directory in directories:
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            photo_filenames.append(path)

    return photo_filenames, sorted(class_names)


def _get_dataset_filename(dataset_dir, split_name, shard_id):
    output_filename = 'sku265_%s_%05d-of-%05d.tfrecord' % (
        split_name, shard_id, _NUM_SHARDS)
    return os.path.join(dataset_dir, output_filename)


def _convert_dataset(split_name, filenames, class_ids, dataset_dir):
    """Converts the given filenames to a TFRecord dataset.

    Args:
      split_name: The name of the dataset, either 'train' or 'validation'.
      filenames: A list of absolute paths to png or jpg images.
      class_ids: A llst of class ids(integer) coorespond to filenames.
      dataset_dir: The directory where the converted datasets are stored.
    """
    assert split_name in ['train', 'validation', 'test']

    num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))

    with tf.Graph().as_default():
        image_reader = ImageReader()

        with tf.Session('') as sess:

            for shard_id in range(_NUM_SHARDS):
                output_filename = _get_dataset_filename(
                    dataset_dir, split_name, shard_id)

                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    start_ndx = shard_id * num_per_shard
                    end_ndx = min((shard_id + 1) * num_per_shard, len(filenames))
                    for i in range(start_ndx, end_ndx):
                        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                            i + 1, len(filenames), shard_id))
                        sys.stdout.flush()

                        # Read the filename:
                        image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()
                        height, width = image_reader.read_image_dims(sess, image_data)

                        class_id = class_ids[i]

                        example = dataset_utils.image_to_tfexample(
                            image_data, b'jpg', height, width, class_id)
                        tfrecord_writer.write(example.SerializeToString())

    sys.stdout.write('\n')
    sys.stdout.flush()


def _clean_up_temporary_files(dataset_dir):
    """Removes temporary files used to create the dataset.

    Args:
      dataset_dir: The directory where the temporary files are stored.
    """
    filename = _DATA_URL.split('/')[-1]
    filepath = os.path.join(dataset_dir, filename)
    tf.gfile.Remove(filepath)

    tmp_dir = os.path.join(dataset_dir, 'flower_photos')
    tf.gfile.DeleteRecursively(tmp_dir)


def _dataset_exists(dataset_dir):
    for split_name in ['train', 'validation']:
        for shard_id in range(_NUM_SHARDS):
            output_filename = _get_dataset_filename(
                dataset_dir, split_name, shard_id)
            if not tf.gfile.Exists(output_filename):
                return False
    return True


def _get_filenames_and_classes_from_label(label_file, image_src_prefix):
    """Get filepath and label from label_file."""
    filenames = []
    class_names = []
    with open(label_file) as fin:
        for line in fin:
            filename, class_name = line.strip().split(' ')
            filenames.append(os.path.join(image_src_prefix, filename))
            class_names.append(class_name)
    return filenames, class_names


def run(dataset_dir, label_file_train, label_file_val, label_file_test, image_src_prefix=''):
    """Runs the download and conversion operation.

    Each line of labels are all of format 'filepath label'.
    Args:
      dataset_dir: The dataset directory where the dataset is stored.
    """
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    if _dataset_exists(dataset_dir):
        print('Dataset files already exist. Exiting without re-creating them.')
        return

    # dataset_utils.download_and_uncompress_tarball(_DATA_URL, dataset_dir)

    train_filenames, train_class_names = _get_filenames_and_classes_from_label(
        label_file_train, image_src_prefix)
    val_filenames, val_class_names = _get_filenames_and_classes_from_label(
        label_file_val, image_src_prefix)
    test_filenames, test_class_names = _get_filenames_and_classes_from_label(
        label_file_test, image_src_prefix)

    train_class_set = set(train_class_names)
    val_class_set = set(val_class_names)
    test_class_set = set(test_class_names)
    assert(not set(train_class_set).difference(val_class_set))
    assert(not set(train_class_set).difference(test_class_set))

    train_class_ids = list(map(int, train_class_names))
    val_class_ids = list(map(int, train_class_names))
    test_class_ids = list(map(int, train_class_names))

    # Divide into train and test:
    random.seed(_RANDOM_SEED)
    inds = [x for x in range(len(train_filenames))]
    random.shuffle(inds)
    train_filenames = [train_filenames[i] for i in inds]
    train_class_ids = [train_class_ids[i] for i in inds]

    # First, convert the training and validation sets.
    _convert_dataset('train', train_filenames, train_class_ids,
                     dataset_dir)
    _convert_dataset('validation', val_filenames, val_class_ids,
                     dataset_dir)
    _convert_dataset('test', test_filenames, test_class_ids,
                     dataset_dir)
    # Finally, write the labels file:
    print('num class {}'.format(len(train_class_set) + 1))
    print('num training data {}'.format(len(train_filenames)))
    print('num validation data {}'.format(len(val_filenames)))
    print('num test data {}'.format(len(test_filenames)))
    print('\nFinished converting the bd_newretales_266 dataset!')


if __name__ == '__main__':
    dst_dir = 'temp'
    label_file = 'imglbl_list'
    run(dst_dir, label_file + '.train', label_file + '.val', label_file + '.test', 'image_data')
