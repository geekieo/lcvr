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


class VisualSimilarNet(BaseModel):
  """similar network only use visual feature
  """

  def create_model(self, 
                   model_input,
                   vocab_size, 
                   num_mixtures=None,
                   l2_penalty=1e-8,):
    """Creates a similar network

    Args:
      model

    """
    layer_1 = slim.fully_connected(
        model_input, 2560, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    layer_1 = tf.nn.l2_normalize(layer_1, 2560)
    layer_2 = slim.fully_connected(
        layer_1, 256, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    layer_2 = tf.nn.l2_normalize(layer_2, 2560)
    return {"predictions": layer_2}






