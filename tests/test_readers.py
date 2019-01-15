# -*- coding: utf-8 -*-
import sys
sys.path.append("..")
import tensorflow as tf
from tensorflow import flags
from tensorflow import gfile
import lcvr.readers as readers
import lcvr.train_utils as train_utils
import lcvr.config as config

FLAGS = flags.FLAGS
# Dataset flags.
flags.DEFINE_string("feature_names", "mean_rgb", "Name of the feature "
                    "to use for training.")
flags.DEFINE_string("feature_sizes", "1024", "Length of the feature vectors.")
# flags.DEFINE_string("feature_sizes", "1500", "Length of the feature vectors.")
flags.DEFINE_bool(
    "frame_features", False,
    "If set, then --train_data_pattern must be frame-level features. "
    "Otherwise, --train_data_pattern must be aggregated video-level "
    "features. The model must also be set appropriately (i.e. to read 3D "
    "batches VS 4D batches.")


def get_reader():
  # Convert feature_names and feature_sizes to lists of values.
  feature_names, feature_sizes = train_utils.GetListOfFeatureNamesAndSizes(
      FLAGS.feature_names, FLAGS.feature_sizes)
  if FLAGS.frame_features:
    reader = readers.YT8MFrameFeatureReader(
        feature_names=feature_names, feature_sizes=feature_sizes)
  else:
    reader = readers.YT8MAggregatedFeatureReader(
        feature_names=feature_names, feature_sizes=feature_sizes)
  return reader

# reader = get_reader()

data_pattern = config.data_pattern
print(data_pattern)
files = gfile.Glob(data_pattern)
num_epochs = 5
filename_queue = tf.train.string_input_producer(
        files, num_epochs=num_epochs, shuffle=True)

batch_size = 1024
reader = readers.YT8MAggregatedFeatureReader(
        feature_names=["mean_rgb"], feature_sizes=["1024"])
reader.prepare_reader(filename_queue,batch_size=batch_size)

num_readers = 2
training_data = [
  reader.prepare_reader(filename_queue,batch_size=1024) for _ in range(num_readers)
]

batch = tf.train.shuffle_batch_join(
        training_data,
        batch_size=batch_size,
        capacity=batch_size * 5,
        min_after_dequeue=batch_size,
        allow_smaller_final_batch=True,
        enqueue_many=True)
unused_video_id, model_input_raw, labels_batch, num_frames = (batch)
print(unused_video_id, model_input_raw, labels_batch, num_frames)

num_towers = 1  #num of copies
tf.summary.histogram("model/input_raw", model_input_raw)
feature_dim = len(model_input_raw.get_shape()) - 1
model_input = tf.nn.l2_normalize(model_input_raw, feature_dim)
tower_inputs = tf.split(model_input, num_towers)
tower_labels = tf.split(labels_batch, num_towers)
tower_num_frames = tf.split(num_frames, num_towers)

#初始化本地变量
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
  sess.run(init_op)
  print(sess.run(model_input))
