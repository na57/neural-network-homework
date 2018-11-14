import numpy as np
import tensorflow as tf 

def flatten(inputs):
  return tf.contrib.layers.flatten(inputs)

def dense(inputs, units, name='dense', activation=tf.nn.relu):
  return tf.layers.dense(inputs, units, name=name, activation=activation)

def conv2d(inputs, num_outputs, kernel_size, activation=tf.nn.relu):
  return tf.contrib.layers.conv2d(inputs, num_outputs, kernel_size,
    activation_fn=activation)

def max_pool2d(inputs, kernel_size):
  return tf.contrib.layers.max_pool2d(inputs, kernel_size)

def build_model(inputs, name='my_model'):
  with tf.variable_scope(name):
    output = inputs
    output = conv2d(output, 8, 3)
    output = max_pool2d(output, 3)
    output = flatten(output)
    # output = dense(output, 16, name='dense_128_1')
    output = dense(output, 32, name='dense_32_1')
    output = dense(output, 16, name='dense_128_1')
    output = dense(output, 8, name='dense_8_1')
    output = dense(output, 2, name='output', activation=tf.nn.softmax)
    return output