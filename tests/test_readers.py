# -*- coding: utf-8 -*-
import sys
sys.path.append("..")
import tensorflow as tf
from tensorflow import flags
from tensorflow import gfile
from tensorflow import logging

FLAGS = flags.FLAGS

# Dataset flags.
flags.DEFINE_string(
    "train_data_pattern", "d:/dataset/yt8m/v2/video/train*.tfrecord",
    "File glob for the training dataset. ")

# Training flags
flags.DEFINE_integer("batch_size",3,
    "How many examples to process per batch for training.")
flags.DEFINE_integer("num_epochs",5,
    "How many passes to make over the dataset before halting training.")

feature_names = ["mean_rgb", "mean_audio"]
num_classes = 3862

def _parse_function(example_proto):
  features = {"id": tf.FixedLenFeature([], tf.string, default_value=""),
              "labels": tf.VarLenFeature(tf.int64)}
  parsed_features = tf.parse_single_example(example_proto, features)
  return parsed_features["id"], parsed_features["labels"]

def train_input_fn():
    filename_list = gfile.Glob(FLAGS.train_data_pattern)
    if not filename_list:
        raise IOError("Unable to find training files. data_pattern='" +
                        data_pattern + "'.")
    print("Number of training files: %s."%str(len(filename_list)))
    # 定义Source
    filenames = tf.placeholder(tf.string, shape=[None])
    dataset = tf.data.TFRecordDataset(filenames)
    # dataset = dataset.shuffle(train_size) # 缓存中的所有数据
    dataset = dataset.map(_parse_function)  # 解析成 tensor
    dataset = dataset.repeat(FLAGS.num_epochs)
    dataset = dataset.batch(FLAGS.batch_size, drop_remainder=True)
    #dataset = dataset.prefetch(1)          # 确保总有一个 batch 准备
    # 消费数据
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()  # tf.Tensor对象
    # # 格式化
    # labels = tf.sparse_to_indicator(next_element["labels"], num_classes) #转换为稠密的布尔指示器张量
    # labels.set_shape([None, num_classes])
    # concatenated_features = tf.concat([
    #     next_element[feature_name] for feature_name in feature_names], 1)
    
    with tf.Session() as sess:
        sess.run(iterator.initializer, feed_dict={filenames: filename_list})
        for i in range(2):
            value = sess.run(next_element)
            assert isinstance(value,tuple)
            print('value: ',value)

if __name__ == "__main__":
    train_input_fn()

