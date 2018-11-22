import numpy as np
import tensorflow as tf 
import datetime

from rice_dataset import *
from tfmodel import *
from utils import *

TRAINING_STEPS = 300
nbatch = 10
MAX_LEFT_PADDING=2
MAX_TOP_PADDING=2

IMAGE_HEIGHT = 224 - MAX_LEFT_PADDING
IMAGE_WIDTH = 224 - MAX_TOP_PADDING

def run():

  # 读入数据集
  x_train, x_test, y_train, y_test = splitted_data()

  # 数据增强
  x_train_aug = []
  y_train_aug = []
  left_padding = []
  top_padding = []
  for i in range(len(x_train)):
    rx,ry,lp,tp = augment(x_train[i], y_train[i], max_left_padding=MAX_LEFT_PADDING, max_top_padding=MAX_TOP_PADDING)
    x_train_aug += rx
    y_train_aug += ry
    left_padding += lp
    top_padding += tp
  x_train = x_train_aug
  y_train = y_train_aug

  print('x_train_aug::', len(x_train_aug))

  # 对label进行独热编码
  y_train = tf.one_hot(y_train, 2)
  y_test = tf.one_hot(y_test, 2)

  # 准备数据流
  iterator = train_dataset_iterator(x_train, y_train, left_padding, top_padding, train_set_length=len(x_train), batch=nbatch)
  iterator_validate = validate_dataset_iterator(x_test, y_test)
  train_item = iterator.get_next()
  validate_item = iterator_validate.get_next()


  # 准备模型
  x = tf.placeholder(tf.float32, shape=(None,IMAGE_HEIGHT,IMAGE_WIDTH,3), name='x-input')
  y = tf.placeholder(tf.float32, shape=(None, 2), name='y-input')


  global_step = tf.Variable(0, trainable=False)
  model = build_model(x)
  y_hat = model
  # 计算交叉熵作为刻画预测值和真实值之间差距的损失函数。
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
    labels=y, logits=y_hat)
  loss = tf.reduce_mean(cross_entropy)
  # 使用tf.train.GradientDescentOptimizer优化算法来优化损失函数。注意这里损失函数包含了交叉熵损失和L2正则化损失。 
  train_op=tf.train.GradientDescentOptimizer(0.001)\
    .minimize(loss, global_step=global_step)

  # 检验神经网络的正确率。
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_hat,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  init = tf.global_variables_initializer()
  with tf.Session() as sess:
    sess.run([iterator.initializer, iterator_validate.initializer, init])

    # 准备验证数据集
    validate_images = sess.run(validate_item)
    validate_feed = {
      x: validate_images[0],
      y: validate_images[1]
    }
    try:
      # 迭代地训练神经网络。
      i = 0
      while True:
      # for i in range(TRAINING_STEPS):
        # 每20轮输出一次在验证数据集上的测试结果。
        if i % 20 == 0:
          loss_val, validate_acc = sess.run([loss, accuracy], feed_dict=validate_feed)
          print("After %d training step(s), validation accuracy "
                          "using average model is %g, loss = %g " % (i, validate_acc, loss_val))
        # 产生这一轮使用的一个batch的训练数据，并运行训练过程。
        x_input, y_input = sess.run(train_item)
        # print('y: ', y_input)
        sess.run(train_op, feed_dict={x: x_input, y: y_input})
        i += 1
    except tf.errors.OutOfRangeError:
      pass


start = datetime.datetime.now()
run()
end = datetime.datetime.now()
print('Done. Elapsed time: ', (end-start).seconds)