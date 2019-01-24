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
flags.DEFINE_string("feature_names", "mean_rgb", 
    "Name of the feature to use for training.")
flags.DEFINE_string("feature_sizes", "1024",
    "Length of the feature vectors.")

# Training flags
flags.DEFINE_integer("batch_size",1024,
    "How many examples to process per batch for training.")
flags.DEFINE_integer("num_epochs",5,
    "How many passes to make over the dataset before halting training.")


files = gfile.Glob( FLAGS.train_data_pattern)
if not files:
      raise IOError("Unable to find training files. data_pattern='" +
                    data_pattern + "'.")
logging.info("Number of training files: %s.", str(len(files)))
filename_queue = tf.train.string_input_producer(
    files, num_epochs=FLAGS.num_epochs)

reader = tf.TFRecordReader()
keys, serialized_examples = reader.read_up_to(filename_queue, num_records=FLAGS.batch_size)

feature_map = {"id": tf.FixedLenFeature([], tf.string),
               "labels": tf.VarLenFeature(tf.int64)}
features = tf.parse_example(serialized_examples, features=feature_map)

# labels = tf.sparse_to_indicator(features["labels"], self.num_classes)
# labels.set_shape([None, self.num_classes])
# batch_data = tf.train.batch(tensors=[features['id'], features['labels']],batch_size=2,dynamic_pad=True)
# concatenated_features = tf.concat([
#     features[feature_name] for feature_name in self.feature_names], 1)

init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
with tf.Session() as sess:
  sess.run(init_op)
  tf.train.start_queue_runners()
  out_features = sess.run(features)
  logging.info(out_features)


# batch_features = tf.train.shuffle_batch_join(
#         features,
#         batch_size=FLAGS.batch_size,
#         capacity=FLAGS.batch_size * 5,
#         min_after_dequeue=FLAGS.batch_size,
#         allow_smaller_final_batch=True,
#         enqueue_many=True)