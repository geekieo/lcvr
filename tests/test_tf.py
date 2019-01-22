# -*- coding: utf-8 -*-
import sys
sys.path.append("..")
import tensorflow as tf
import logger

logging =  logger.Logging(name='test_tf', filename='log')

logging.info('!!!!!!!')

a = tf.Variable('aaa')
b = tf.Variable('bbb')

logging.info(str(a)+str(b))
# logging.info(type(a))

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
  sess.run(init_op)
  logging.info(sess.run(a).decode())
  logging.info(sess.run(b))