# neural-network-homework
2018年神经网络课程作业

## 第一次模型实验

网络结构：
```python
def build_model(inputs, name='my_model'):
  with tf.variable_scope(name):
    output = flatten(inputs)
    output = dense(output, 32, name='dense_32_1')
    output = dense(output, 16, name='dense_16 _1')
    output = dense(output, 8, name='dense_8_1')
    output = dense(output, 2, name='output', activation=tf.nn.softmax)
    return output
```

测试精度：0.73