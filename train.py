# -*- coding: utf-8 -*-
import os
import json
import time

import tensorflow as tf
from tensorflow import app
from tensorflow import flags
from tensorflow import gfile
from tensorflow.python.lib.io import file_io

import losses
import readers
import models
import logger
import utils
import export_model

logging = logger.Logging('train','lcvr')
FLAGS = flags.FLAGS

if __name__ == '__main__':
  # Dataset flags.
  flags.DEFINE_string("train_dir", "/tmp/yt8m_model/",
                      "The directory to save the model files in.")
  flags.DEFINE_string(
      "train_data_pattern", "/data/dataset/yt8m/v2/video/train*.tfrecord",
      "File glob for the training dataset. If the files refer to Frame Level "
      "features (i.e. tensorflow.SequenceExample), then set --reader_type "
      "format. The (Sequence)Examples are expected to have 'rgb' byte array "
      "sequence feature as well as a 'labels' int64 context feature.")
  flags.DEFINE_string("feature_names", "mean_rgb", "Name of the feature "
                      "to use for training.")
  flags.DEFINE_string("feature_sizes", "1024", "Length of the feature vectors.")

  # Model flags.
  flags.DEFINE_bool(
      "frame_features", False,
      "If set, then --train_data_pattern must be frame-level features. "
      "Otherwise, --train_data_pattern must be aggregated video-level "
      "features. The model must also be set appropriately (i.e. to read 3D "
      "batches VS 4D batches.")
  flags.DEFINE_string(
      "model", "LogisticModel",
      "Which architecture to use for the model. Models are defined "
      "in models.py.")
  flags.DEFINE_bool(
      "start_new_model", False,
      "If set, this will not resume from a checkpoint and will instead create a"
      " new model instance.")

  # Training flags.
  flags.DEFINE_integer("max_steps", None,
                       "The maximum number of iterations of the training loop.")
  flags.DEFINE_integer("export_model_steps", 1000,
                       "The period, in number of steps, with which the model "
                       "is exported for batch prediction.")
  flags.DEFINE_string("label_loss", "CrossEntropyLoss",
                      "Which loss function to use for training the model.")
  flags.DEFINE_string("optimizer", "AdamOptimizer",
                      "What optimizer class to use.")

  # Other flags.   
  flags.DEFINE_bool(
      "log_device_placement", False,
      "Whether to write the device on which every op will run into the "
      "logs on startup.")  


def get_reader():
  # Convert feature_names and feature_sizes to lists of values.
  feature_names, feature_sizes = utils.GetListOfFeatureNamesAndSizes(
      FLAGS.feature_names, FLAGS.feature_sizes)

  if FLAGS.frame_features:
    reader = readers.YT8MFrameFeatureReader(
        feature_names=feature_names, feature_sizes=feature_sizes)
  else:
    reader = readers.YT8MAggregatedFeatureReader(
        feature_names=feature_names, feature_sizes=feature_sizes)

  return reader


def find_class_by_name(name, modules):
  """Searches the provided modules for the named class and returns it."""
  modules = [getattr(module, name, None) for module in modules]
  return next(a for a in modules if a)


def get_input_data_tensors(reader,
                           data_pattern,
                           batch_size=1000,
                           num_epochs=None,
                           num_readers=1):
  """Creates the section of the graph which reads the training data.

  Args:
    reader: A class which parses the training data.
    data_pattern: A 'glob' style path to the data files.
    batch_size: How many examples to process at a time.
    num_epochs: How many passes to make over the training data. Set to 'None'
                to run indefinitely.
    num_readers: How many I/O threads to use.

  Returns:
    A tuple containing the features tensor, labels tensor, and optionally a
    tensor containing the number of frames per video. The exact dimensions
    depend on the reader being used.

  Raises:
    IOError: If no files matching the given pattern were found.
  """
  logging.info("Using batch size of " + str(batch_size) + " for training.")
  with tf.name_scope("train_input"):
    files = gfile.Glob(data_pattern)
    if not files:
      raise IOError("Unable to find training files. data_pattern='" +
                    data_pattern + "'.")
    logging.info("Number of training files: %s.", str(len(files)))
    filename_queue = tf.train.string_input_producer(
        files, num_epochs=num_epochs, shuffle=True)
    training_data = [
        reader.prepare_reader(filename_queue) for _ in range(num_readers)
    ]

    return tf.train.shuffle_batch_join(
        training_data,
        batch_size=batch_size,
        capacity=batch_size * 5,
        min_after_dequeue=batch_size,
        allow_smaller_final_batch=True,
        enqueue_many=True)


def build_graph(reader,
                model,
                train_data_pattern,
                label_loss_fn=losses.CrossEntropyLoss(),
                batch_size=1000,
                base_learning_rate=0.01,
                learning_rate_decay_examples=1000000,
                learning_rate_decay=0.95,
                optimizer_class=tf.train.AdamOptimizer,
                clip_gradient_norm=1.0,
                regularization_penalty=1,
                num_readers=1,
                num_epochs=None):
  """Creates the Tensorflow graph.

  This will only be called once in the life of
  a training model, because after the graph is created the model will be
  restored from a meta graph file rather than being recreated.

  Args:
    reader: The data file reader. It should inherit from BaseReader.
    model: The core model (e.g. logistic or neural net). It should inherit
           from BaseModel.
    train_data_pattern: glob path to the training data files.
    label_loss_fn: What kind of loss to apply to the model. It should inherit
                from BaseLoss.
    batch_size: How many examples to process at a time.
    base_learning_rate: What learning rate to initialize the optimizer with.
    optimizer_class: Which optimization algorithm to use.
    clip_gradient_norm: Magnitude of the gradient to clip to.
    regularization_penalty: How much weight to give the regularization loss
                            compared to the label loss.
    num_readers: How many threads to use for I/O operations.
    num_epochs: How many passes to make over the data. 'None' means an
                unlimited number of passes.
  """

  global_step = tf.Variable(0, trainable=False, name="global_step")

  local_device_protos = device_lib.list_local_devices()
  gpus = [x.name for x in local_device_protos if x.device_type == 'GPU']
  gpus = gpus[:FLAGS.num_gpu]
  num_gpus = len(gpus)

  if num_gpus > 0:
    logging.info("Using the following GPUs to train: " + str(gpus))
    num_towers = num_gpus
    device_string = '/gpu:%d'
  else:
    logging.info("No GPUs found. Training on CPU.")
    num_towers = 1
    device_string = '/cpu:%d'

  learning_rate = tf.train.exponential_decay(
      base_learning_rate,
      global_step * batch_size * num_towers,
      learning_rate_decay_examples,
      learning_rate_decay,
      staircase=True)
  tf.summary.scalar('learning_rate', learning_rate)

  optimizer = optimizer_class(learning_rate)
  unused_video_id, model_input_raw, labels_batch, num_frames = (
      get_input_data_tensors(
          reader,
          train_data_pattern,
          batch_size=batch_size * num_towers,
          num_readers=num_readers,
          num_epochs=num_epochs))
  tf.summary.histogram("model/input_raw", model_input_raw)

  feature_dim = len(model_input_raw.get_shape()) - 1

  model_input = tf.nn.l2_normalize(model_input_raw, feature_dim)

  tower_inputs = tf.split(model_input, num_towers)
  tower_labels = tf.split(labels_batch, num_towers)
  tower_num_frames = tf.split(num_frames, num_towers)
  tower_gradients = []
  tower_predictions = []
  tower_label_losses = []
  tower_reg_losses = []
  for i in range(num_towers):
    # For some reason these 'with' statements can't be combined onto the same
    # line. They have to be nested.
    with tf.device(device_string % i):
      with (tf.variable_scope(("tower"), reuse=True if i > 0 else None)):
        with (slim.arg_scope([slim.model_variable, slim.variable], device="/cpu:0" if num_gpus!=1 else "/gpu:0")):
          result = model.create_model(
            tower_inputs[i],
            num_frames=tower_num_frames[i],
            vocab_size=reader.num_classes,
            labels=tower_labels[i])
          for variable in slim.get_model_variables():
            tf.summary.histogram(variable.op.name, variable)

          predictions = result["predictions"]
          tower_predictions.append(predictions)

          if "loss" in result.keys():
            label_loss = result["loss"]
          else:
            label_loss = label_loss_fn.calculate_loss(predictions, tower_labels[i])

          if "regularization_loss" in result.keys():
            reg_loss = result["regularization_loss"]
          else:
            reg_loss = tf.constant(0.0)

          reg_losses = tf.losses.get_regularization_losses()
          if reg_losses:
            reg_loss += tf.add_n(reg_losses)

          tower_reg_losses.append(reg_loss)

          # Adds update_ops (e.g., moving average updates in batch normalization) as
          # a dependency to the train_op.
          update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
          if "update_ops" in result.keys():
            update_ops += result["update_ops"]
          if update_ops:
            with tf.control_dependencies(update_ops):
              barrier = tf.no_op(name="gradient_barrier")
              with tf.control_dependencies([barrier]):
                label_loss = tf.identity(label_loss)

          tower_label_losses.append(label_loss)

          # Incorporate the L2 weight penalties etc.
          final_loss = regularization_penalty * reg_loss + label_loss
          gradients = optimizer.compute_gradients(final_loss,
              colocate_gradients_with_ops=False)
          tower_gradients.append(gradients)
  label_loss = tf.reduce_mean(tf.stack(tower_label_losses))
  tf.summary.scalar("label_loss", label_loss)
  if regularization_penalty != 0:
    reg_loss = tf.reduce_mean(tf.stack(tower_reg_losses))
    tf.summary.scalar("reg_loss", reg_loss)
  merged_gradients = utils.combine_gradients(tower_gradients)

  if clip_gradient_norm > 0:
    with tf.name_scope('clip_grads'):
      merged_gradients = utils.clip_gradient_norms(merged_gradients, clip_gradient_norm)

  train_op = optimizer.apply_gradients(merged_gradients, global_step=global_step)

  tf.add_to_collection("global_step", global_step)
  tf.add_to_collection("loss", label_loss)
  tf.add_to_collection("predictions", tf.concat(tower_predictions, 0))
  tf.add_to_collection("input_batch_raw", model_input_raw)
  tf.add_to_collection("input_batch", model_input)
  tf.add_to_collection("num_frames", num_frames)
  tf.add_to_collection("labels", tf.cast(labels_batch, tf.float32))
  tf.add_to_collection("train_op", train_op)


def start_server(cluster, task):
    """Creates a Server.

    Args:
      cluster: A tf.train.ClusterSpec if the execution is distributed.
        None otherwise.
      task: A TaskSpec describing the job type and the task index.
    """

    if not task.type:
      raise ValueError("%s: The task type must be specified." %
                      task_as_string(task))
    if task.index is None:
      raise ValueError("%s: The task index must be specified." %
                      task_as_string(task))

    # Create and start a server.
    return tf.train.Server(
        tf.train.ClusterSpec(cluster),
        protocol="grpc",
        job_name=task.type,
        task_index=task.index)


def task_as_string(task):
  return "/job:%s/task:%s" % (task.type, task.index)


class Trainer(object):
  """A Trainer to train a Tensorflow graph."""

  def __init__(self, cluster, task, train_dir, model, reader, model_exporter,
               log_device_placement=True, max_steps=None,
               export_model_steps=1000):
    """"Creates a Trainer.

    Args:
      cluster: A tf.train.ClusterSpec if the execution is distributed.
        None otherwise.
      task: A TaskSpec describing the job type and the task index.
    """
    self.cluster = cluster
    self.task = task
    self.is_master = (task.type == "master" and task.index == 0)
    self.train_dir = train_dir
    self.config = tf.ConfigProto(
        allow_soft_placement=True,log_device_placement=log_device_placement)
    self.model = model
    self.reader = reader
    self.model_exporter = model_exporter
    self.max_steps = max_steps
    self.max_steps_reached = False
    self.export_model_steps = export_model_steps
    self.last_model_export_step = 0
  
  def remove_training_directory(self, train_dir):
    """Removes the training directory."""
    try:
      logging.info(
          "%s: Removing existing train directory.",
          task_as_string(self.task))
      gfile.DeleteRecursively(train_dir)
    except:
      logging.error(
          "%s: Failed to delete directory " + train_dir +
          " when starting a new model. Please delete it manually and" +
          " try again.", task_as_string(self.task))

  def start_server_if_distributed(self):
    """Starts a server if the execution is distributed."""
    if self.cluster:
      logging.info("%s: Starting trainer within cluster %s.",
                   task_as_string(self.task), self.cluster.as_dict())
      server = start_server(self.cluster, self.task)
      target = server.target
      device_fn = tf.train.replica_device_setter(
          ps_device="/job:ps",
          worker_device="/job:%s/task:%d" % (self.task.type, self.task.index),
          cluster=self.cluster)
    else:
      target = ""
      device_fn = ""
    return (target, device_fn)

  def get_meta_filename(self, start_new_model, train_dir):
    if start_new_model:
      logging.info("%s: Flag 'start_new_model' is set. Building a new model.",
                   task_as_string(self.task))
      return None
    latest_checkpoint = tf.train.latest_checkpoint(train_dir)
    if not latest_checkpoint:
      logging.info("%s: No checkpoint file found. Building a new model.",
                   task_as_string(self.task))
      return None
    meta_filename = latest_checkpoint + ".meta"
    if not gfile.Exists(meta_filename):
      logging.info("%s: No meta graph file found. Building a new model.",
                     task_as_string(self.task))
      return None
    else:
      return meta_filename

  def recover_model(self, meta_filename):
    logging.info("%s: Restoring from meta graph file %s",
                 task_as_string(self.task), meta_filename)
    return tf.train.import_meta_graph(meta_filename)

  def build_model(self, model, reader):
    """Find the model and build the graph."""

    label_loss_fn = find_class_by_name(FLAGS.label_loss, [losses])()
    optimizer_class = find_class_by_name(FLAGS.optimizer, [tf.train])

    build_graph(reader=reader,
                model=model,
                optimizer_class=optimizer_class,
                clip_gradient_norm=FLAGS.clip_gradient_norm,
                train_data_pattern=FLAGS.train_data_pattern,
                label_loss_fn=label_loss_fn,
                base_learning_rate=FLAGS.base_learning_rate,
                learning_rate_decay=FLAGS.learning_rate_decay,
                learning_rate_decay_examples=FLAGS.learning_rate_decay_examples,
                regularization_penalty=FLAGS.regularization_penalty,
                num_readers=FLAGS.num_readers,
                batch_size=FLAGS.batch_size,
                num_epochs=FLAGS.num_epochs)

    return tf.train.Saver(max_to_keep=0, keep_checkpoint_every_n_hours=0.25)

  def run(self, start_new_model=False):
    """Performs training on the currently defined Tensorflow graph.

    Returns:
      A tuple of the training Hit@1 and the training PERR.
    """
    if self.is_master and start_new_model:
      self.remove_training_directory(self.train_dir)

    if not os.path.exists(self.train_dir):
      os.makedirs(self.train_dir)

    model_flags_dict = {
        "model": FLAGS.model,
        "feature_sizes": FLAGS.feature_sizes,
        "feature_names": FLAGS.feature_names,
        "frame_features": FLAGS.frame_features,
        "label_loss": FLAGS.label_loss,
    }

    # write flags to a json file, or load existing flags from json file.
    flags_json_path = os.path.join(FLAGS.train_dir, "model_flags.json")
    if file_io.file_exists(flags_json_path):
      existing_flags = json.load(file_io.FileIO(flags_json_path, mode="r"))
      if existing_flags != model_flags_dict:
        logging.error("Model flags do not match existing file %s.Please "
                      "delete the file, change --train_dir, or pass flag "
                      "--start_new_model",
                      flags_json_path)
        logging.error("Ran model with flags: %s", str(model_flags_dict))
        logging.error("Previously ran with flags: %s", str(existing_flags))
        exit(1)
    else:
      with file_io.FileIO(flags_json_path, mode='w') as fout:
        fout.write(json.dumps(model_flags_dict))
    
    # if not distributed, return ['','']
    target, device_fn = self.start_server_if_distributed()

    # if meta file exist, return meta filename for continue training
    meta_filename = self.get_meta_filename(start_new_model, self.train_dir)

    with tf.Graph().as_default() as graph:
      if meta_filename:
        saver = self.recover_model(meta_filename)

      with tf.device(device_fn):
        if not meta_filename:
          saver = self.build_model(self.model, self.reader)

        global_step = tf.get_collection("global_step")[0]
        loss = tf.get_collection("loss")[0]
        predictions = tf.get_collection("predictions")[0]
        labels = tf.get_collection("labels")[0]
        train_op = tf.get_collection("train_op")[0]
        init_op = tf.global_variables_initializer()

    # a training helper that checkpoints models and computes summaries.
    supervisor = tf.train.Supervisor(
        graph,
        logdir=self.train_dir,
        init_op=init_op,
        is_chief=self.is_master,
        global_step=global_step,
        save_model_secs=15 * 60,
        save_summaries_secs=120,
        saver=saver)

    logging.info("%s: Starting managed session.", task_as_string(self.task))
    with supervisor.managed_session(target, config=self.config) as sess:
      pass


def main(unused_argv):
  logging = logger.Logging(name='train',filename='lcvr')
  logging.info("Tensorflow version: %s.",tf.__version__)
  
  model = find_class_by_name(FLAGS.model, [models])()
  reader = get_reader()
  model_exporter = export_model.ModelExporter(
      frame_features=FLAGS.frame_features, model=model, reader=reader)
  
  cluster = None
  task_data = {"type": "master", "index": 0}
  task = type("TaskSpec", (object,), task_data)
  trainer = Trainer(cluster, task, FLAGS.train_dir, model, reader, model_exporter,
          FLAGS.log_device_placement, FLAGS.max_steps,
          FLAGS.export_model_steps)
  trainer.run(start_new_model=FLAGS.start_new_model)  

if __name__ == "__main__":
  app.run()
