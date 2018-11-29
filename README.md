# neural-network-homework
2018年神经网络课程作业

## 实验 #1

- 网络结构：
```python
def build_model(inputs, name='my_model'):
  with tf.variable_scope(name):
    output = flatten(inputs)
    output = dense(output, 32, name='dense_32_1')
    output = dense(output, 16, name='dense_16_1')
    output = dense(output, 8, name='dense_8_1')
    output = dense(output, 2, name='output', activation=tf.nn.softmax)
    return output
```
- 主要参数：
  - lr: 0.01
  - batch size: 10
  - loss function: tf.nn.softmax_cross_entropy_with_logits_v2
  - cost function: L2
- 精度：0.73
- 执行时间：166s

## 实验 #2

- 网络结构：
```python
def build_model(inputs, name='my_model'):
  with tf.variable_scope(name):
    output = inputs
    output = conv2d(output, 8, 3)
    output = max_pool2d(output, 3)
    output = flatten(output)
    output = dense(output, 32, name='dense_32_1')
    output = dense(output, 16, name='dense_128_1')
    output = dense(output, 8, name='dense_8_1')
    output = dense(output, 2, name='output', activation=tf.nn.softmax)
    return output
```
- 主要参数：
  - lr: 0.01
  - batch size: 10
  - loss function: tf.nn.softmax_cross_entropy_with_logits_v2
  - cost function: L2
- 精度：0.75
- 执行时间：166s
- 结论：使用卷积之后，精度略有上升，但效果不明显

## 实验 #3

- 网络结构：
```python
def build_model(inputs, name='my_model'):
  with tf.variable_scope(name):
    output = inputs
    output = flatten(output)
    output = dense(output, 16, name='dense_128_1')
    output = dense(output, 2, name='output', activation=tf.nn.softmax)
    return output
```
- 主要参数：
  - lr: 0.001
  - batch size: 10
  - loss function: 交叉熵
  - cost function: L2
- 精度：0.76
- 执行时间：316s
- 样本数量：80k
- 结论：样本数量增加，网络简单，精度有变化，但不稳定。


# 实验 #4 （以下实验使用单个GPU）

- 网络结构：
```python
def build_model(inputs, name='my_model'):
  with tf.variable_scope(name):
    output = inputs

    output = conv2d(output, 64, 3)
    output = max_pool2d(output, 2)
    
    output = conv2d(output, 128, 3)
    output = max_pool2d(output, 2)
    
    output = conv2d(output, 256, 3)
    output = conv2d(output, 256, 3)
    output = max_pool2d(output, 2)
    
    output = conv2d(output, 512, 3)
    output = conv2d(output, 512, 3)
    output = max_pool2d(output, 2)
    
    output = conv2d(output, 512, 3)
    output = conv2d(output, 512, 3)
    output = max_pool2d(output, 2)

    output = flatten(output)

    output = dense(output, 4096, name='dense_32_1')
    output = dense(output, 4096, name='dense_128_1')
    output - dropout(output)

    output = dense(output, 2, name='output', activation=tf.nn.softmax)
    return output
```

- 主要参数：
  - lr: 0.001/0.01
  - batch size: 64
  - loss function: 交叉熵
  - cost function: L2
- 精度：0.52/0.74
- 执行时间：1016s
- 样本数量：80k
- 结论：第一次实验梯度下降太慢，几乎不下降，后来发现lr太小，所以更改了lr再次做了实验。


## 实验 train-vgg16-1127
使用了VGG16的网络架构，其他参数不变，实验发现损失还是无法下降，精度在0.6左右徘徊。因为网络复杂，会产生内存溢出的错误，因此，之后的实验在测试的时候只取20个样本。

## 实验 train-vgg16-1127#2
将优化方法改为动量方法，lr=0.01, 动量为0.9，参数与vgg16一致，但max_pool层stride=2。梯度明细有所下降，最终下降到0.55左右，精度最高可达0.8，但也不太稳定。

## 实验 #
改动：
- 所有stride=1

结论：程序无法运行，报服务器内存溢出错误。debug后发现，是在run tf.global_variables_initializer()时引起的错误，初步判定是由于网络参数太多导致的。

## 实验 #train-vgg16-20181128-0853

改动：
- epoch：3
- 前两个卷积层增加了batch_norm

结论：
- 前阶段损失下降非常快，后面无法下降，反而上升，出现了退化问题
- 目前GPU只能支持到前两个卷积增加batch_norm，后面再增加会出现OOM问题

## 实验 #train-vgg16-20181129-0331
改动
- max_left_padding = 10, max_top_padding = 10

结论：
- 损失无法下降

## 实验 # train-vgg16-20181129-0421
改动：
- 训练数据集的shuffe_buffer由100改为10000

结论：
- 大约在第1200次循环的时候损失达到最低，精度在0.75左右，然后开始上升，直到1450次循环后损失达到最大，并饱和。通过滑动窗口进行的数据增强似乎没有效果。

## 实验 # train-vgg16-20181129-0520
改动：
- 使用batch_norm对输入数据进行归一化.

结论：
- 下降很快，大约在700次循环后损失达到最小，精度可达0.8
- 后期损失回升，同样出现退化现象

# 实验 # train-vgg16-20181129-0557
改动：
- 不再使用数据增强，改为增加轮次：epoch=10
结论：
- 计算时间4355s
- 目前为止，损失下降效果最好的一次。损失一直在下降，约2000次循环后下降减慢。
- 精度基本可以在0.75以上，平均大约0.8左右

# 实验 #
改动：
- 继续增大梯度 epoch=30
- 每2000次循环lr后减小10倍。