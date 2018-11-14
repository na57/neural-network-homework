import numpy as np
import tensorflow as tf 

def flatten(inputs):
  return tf.contrib.layers.flatten(inputs)

def dense(inputs, units, name='dense', activation=tf.nn.relu):
  return tf.layers.dense(inputs, units, name=name, activation=activation)

def build_model(inputs, name='model'):
  with tf.variable_scope(name):
    f_1 = flatten(inputs)
    d_2 = dense(f_1, 16, name='dense_2')
    d_3 = dense(d_2, 32, name='dense_3')
    output = dense(d_3, 2, name='output', activation=tf.nn.softmax)
    return output