# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.contrib.slim as slim

class BaseModel(object):
  """Inherit from this class when implementing new models."""

  def create_model(self, unused_model_input, **unused_params):
    raise NotImplementedError()


class LogisticModel(BaseModel):
  """Logistic model with L2 regularization."""

  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
    """Creates a logistic model.

    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""
    output = slim.fully_connected(
        model_input, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    return {"predictions": output}


class DeepNeuralNet(BaseModel):
  """a shallow deep neural network"""

  def create_model(self, model_input, vocab_size, l2_penalty=1e-8,**unused_params):
    """Creates a similar network

    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""
    model_input = tf.nn.l2_normalize(model_input)
    layer_1 = slim.fully_connected(
        model_input, 2560, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    layer_2 = slim.fully_connected(
        layer_1, 256, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    layer_2 = tf.nn.l2_normalize(layer_2)
    output = slim.fully_connected(
        layer_2, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    return {"predictions": output}

class VisualNet(BaseModel):
  """embedding network of visual input"""

  def create_model(self, model_input, vocab_size, l2_penalty=1e-8,**unused_params):
    """Creates a embedding network only use visual features

    Args:
      model_input: 'batch' x 'num_features' matrix of input features.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'embeddings' key. The dimensions of the tensor are
      batch_size x number of nodes in output layer."""
    model_input = tf.nn.l2_normalize(model_input)
    layer_1 = slim.fully_connected(
        model_input, 2560, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    layer_2 = slim.fully_connected(
        layer_1, 256, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    output = tf.nn.l2_normalize(layer_2)
    return {"predictions": output}

