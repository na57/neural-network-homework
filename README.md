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