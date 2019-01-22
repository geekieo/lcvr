# -*- coding: utf-8 -*-
import sys
sys.path.append("..")
import tensorflow as tf
from tensorflow import flags
from tensorflow import gfile

import readers
import logger

logging = logger.Logging()

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
flags.DEFINE_string(
      "train_data_pattern", "d:/dataset/yt8m/v2/video/train*.tfrecord",
      "File glob for the training dataset. If the files refer to Frame Level "
      "features (i.e. tensorflow.SequenceExample), then set --reader_type "
      "format. The (Sequence)Examples are expected to have 'rgb' byte array "
      "sequence feature as well as a 'labels' int64 context feature.")


files = gfile.Glob( FLAGS.train_data_pattern)
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
logging.info('!!!!!!!')
logging.info(str(unused_video_id)+str(model_input_raw)+str(labels_batch)+str(num_frames))

num_towers = 1  #num of copies
tf.summary.histogram("model/input_raw", model_input_raw)
feature_dim = len(model_input_raw.get_shape()) - 1
model_input = tf.nn.l2_normalize(model_input_raw, feature_dim)
tower_inputs = tf.split(model_input, num_towers)
tower_labels = tf.split(labels_batch, num_towers)
tower_num_frames = tf.split(num_frames, num_towers)
logging.info(str(tower_inputs)+str(tower_labels)+str(tower_num_frames))

#初始化本地变量
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
  sess.run(init_op)
  logging.info(sess.run(model_input[0]))
