import numpy as np
import tensorflow as tf 

def flatten(inputs):
  return tf.contrib.layers.flatten(inputs)

def dense(inputs, units, name='dense', activation=tf.nn.relu):
  return tf.layers.dense(inputs, units, name=name, activation=activation)

def build_model(inputs, name='my_model'):
  with tf.variable_scope(name):
    output = flatten(inputs)
    # output = dense(output, 16, name='dense_128_1')
    output = dense(output, 32, name='dense_32_1')
    output = dense(output, 16, name='dense_128_1')
    output = dense(output, 8, name='dense_8_1')
    output = dense(output, 2, name='output', activation=tf.nn.softmax)
    return output