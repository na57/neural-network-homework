import numpy as np
import tensorflow as tf 
import datetime
import os

from rice_dataset import *
from tfmodel import *
from utils import *

# 记录开始时间
start = datetime.datetime.now()

# 设定日志输出等级
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

TRAINING_STEPS = 300
nbatch = 2
nepoch=3
MAX_LEFT_PADDING=2
MAX_TOP_PADDING=2

IMAGE_HEIGHT = 224 - MAX_LEFT_PADDING
IMAGE_WIDTH = 224 - MAX_TOP_PADDING

summaries_dir = './summaries'
train_name = 'train-vgg16-' + datetime.datetime.today().strftime('%Y%m%d-%H%M')
validate_name = 'validate-vgg16-' + datetime.datetime.today().strftime('%Y%m%d-%H%M')


# 验证

# def accuracy(y, y_hat):
#   correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_hat,1))
#   accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#   return accuracy

def validate(x_test, y_test, x, y, y_hat, train_steps):
  iterator_validate = validate_dataset_iterator(x_test, y_test, batch=nbatch)
  validate_item = iterator_validate.get_next()

  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_hat,1))
  validate_correct_prediction = tf.Variable([], 'validate-correct-prediction')
  validate_accuracy = tf.reduce_mean(tf.cast(validate_correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', validate_accuracy)

  init = tf.global_variables_initializer()
  with tf.Session() as sess:
    sess.run([iterator_validate.initializer, init])

    # 写到指定的磁盘路径中
    validate_writer = tf.summary.FileWriter(os.path.join(summaries_dir, validate_name), sess.graph)
    try:
      while True:
        x_input_validate, y_input_validate = sess.run(validate_item)
        batch_correct_prediction = tf.Variable(sess.run(correct_prediction, feed_dict={
          x: x_input_validate,
          y: y_input_validate
        }))
        validate_correct_prediction.assign(tf.concat([validate_correct_prediction, batch_correct_prediction], 0))
    except tf.errors.OutOfRangeError:
      pass
    validate_accuracy_val = sess.run(validate_accuracy)
    # validate_writer.add_summary(validate_accuracy_val, global_step=train_steps)

    print("After %d training step(s), validation accuracy "
                    "using average model is %g" % (train_steps, validate_accuracy_val))



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
  iterator = train_dataset_iterator(x_train, y_train, left_padding, top_padding, train_set_length=len(x_train),
                                    batch=nbatch, epoch=nepoch)
  train_item = iterator.get_next()


  # 准备模型
  x = tf.placeholder(tf.float32, shape=(None,IMAGE_HEIGHT,IMAGE_WIDTH,3), name='x-input')
  y = tf.placeholder(tf.float32, shape=(None, 2), name='y-input')
  y_hat = tf.placeholder(tf.float32, shape=(None, 2), name='y-output')

  global_step = tf.Variable(0, trainable=False)
  y_hat = build_model(x)

  # 计算交叉熵作为刻画预测值和真实值之间差距的损失函数。
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
    labels=y, logits=y_hat)
  loss = tf.reduce_mean(cross_entropy)
  tf.summary.scalar('loss', loss)

  train_op=tf.train.MomentumOptimizer(0.01, 0.9)\
    .minimize(loss, global_step=global_step)

  init = tf.global_variables_initializer()
 

  with tf.Session() as sess:
    sess.run([ iterator.initializer, init])

    # 写到指定的磁盘路径中
    train_writer = tf.summary.FileWriter(os.path.join(summaries_dir, train_name), sess.graph)
    # summaries合并
    merged = tf.summary.merge_all()

    try:
      # 迭代地训练神经网络。
      i = 0
      while True:
      # for i in range(TRAINING_STEPS):
        # 每20轮输出一次在验证数据集上的测试结果。
        if i % 20 == 0:
          validate(x_test,y_test,x,y,y_hat,i)
        # 产生这一轮使用的一个batch的训练数据，并运行训练过程。
        x_input, y_input = sess.run(train_item)
        
        merged_summary, _ = sess.run([merged, train_op], feed_dict={x: x_input, y: y_input})
        train_writer.add_summary(merged_summary, global_step=i)
        i += 1
    except tf.errors.OutOfRangeError:
      pass

run()
end = datetime.datetime.now()
print('Done. Elapsed time: ', (end-start).seconds)