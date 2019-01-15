# -*- coding: utf-8 -*-
import tensorflow.contrib.slim as slim

class BaseModel(object):
  """Inherit from this class when implementing new models."""

  def create_model(self, unused_model_input, **unused_params):
    raise NotImplementedError()


class VisualSimilarNet(BaseModel):
  """similar network only use visual feature
  """

  def create_model(self, 
                   model_input,
                   num_mixtures=None,
                   l2_penalty=1e-8,):
    layer_1 = slim.fully_connected(
        model_input, 2560, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    layer_1 = tf.nn.l2_normalize(layer_1, 2560)
    layer_2 = slim.fully_connected(
        layer_1, 256, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    layer_2 = tf.nn.l2_normalize(layer_2, 2560)
    return {"predictions": layer_2}




