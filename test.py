import numpy as np
import tensorflow as tf 

from rice_dataset import *
from tfmodel import *

TRAINING_STEPS = 300
nbatch = 10


def run():

  # 准备数据流
  x_train, x_test, y_train, y_test = splitted_data()
  y_train = tf.one_hot(y_train, 2)
  y_test = tf.one_hot(y_test, 2)
  iterator = train_dataset_iterator(x_train, y_train, batch=nbatch)
  iterator_validate = validate_dataset_iterator(x_test, y_test)
  train_item = iterator.get_next()
  validate_item = iterator_validate.get_next()


  # 准备模型
  x = tf.placeholder(tf.float32, shape=(None,224,224,3), name='x-input')
  y = tf.placeholder(tf.float32, shape=(None, 2), name='y-input')


  global_step = tf.Variable(0, trainable=False)
  model = build_model(x)
  y_hat = model
  # 计算交叉熵作为刻画预测值和真实值之间差距的损失函数。
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
    labels=y, logits=y_hat)
  loss = tf.reduce_mean(cross_entropy)
  # 使用tf.train.GradientDescentOptimizer优化算法来优化损失函数。注意这里损失函数包含了交叉熵损失和L2正则化损失。 
  train_op=tf.train.GradientDescentOptimizer(0.01)\
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
        # 每1000轮输出一次在验证数据集上的测试结果。
        if i % 20 == 0:
          validate_acc = sess.run(accuracy, feed_dict=validate_feed)
          print("After %d training step(s), validation accuracy "
                          "using average model is %g " % (i, validate_acc))
        # 产生这一轮使用的一个batch的训练数据，并运行训练过程。
        image = sess.run(train_item)
        sess.run(train_op, feed_dict={x: image[0], y: image[1]})
        i += 1
    except tf.errors.OutOfRangeError:
      print('Done')
      pass


run()
