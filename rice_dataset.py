'''
Rice数据集。
图片默认文件夹：../datasets/refine_RiceDataset

- allData 返回所有数据
    - 返回值：(x, y)
        x: 文件路径
        y: 分类
- splitted_data 获取划分好的数据和标签
    - 返回值：(x_train, x_test, y_train, y_test)
- read_image_tensor 读取图片数据tensor
- image_tensor_norm 对图片数据tensor进行规范化
'''

import numpy as np
import os
import glob
import tensorflow as tf 

from sklearn.model_selection import train_test_split

SOUND = 1  # 完善粒
UNSOUND = 0 # 不完善粒

# 加载数据集
data_dir=os.path.join('..', 'datasets','refine_RiceDataset')
sound_dir = os.path.join(data_dir,'**',str(SOUND),'**','*.jpg')
unsound_dir = os.path.join(data_dir,'**',str(UNSOUND),'**','*.jpg')

def soundData(data_dir=sound_dir):
    path_sound = glob.glob(sound_dir, recursive=True)
    print('sound examples: ', len(path_sound))
    sound_labels = np.ones(len(path_sound)).astype(int)
    return path_sound, sound_labels

def unsoundData(data_dir=unsound_dir):
    path_unsound = glob.glob(unsound_dir, recursive=True)
    print('unsound examples: ', len(path_unsound))
    unsound_labels = np.zeros(len(path_unsound)).astype(int)
    return path_unsound, unsound_labels
    
def allData():
    x_sound, y_sound= soundData()
    x_unsound, y_unsound = unsoundData()
    x = np.append(x_sound, x_unsound)
    y = np.append(y_sound, y_unsound)
    return x, y

# 划分训练集和测试集
def splitted_data(**kw):
    x, y = allData()
    x_train, x_test, y_train, y_test = train_test_split(x, y, **kw)
    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)
    return x_train, x_test, y_train, y_test

'''
根据路径读取图片数据
'''
def read_image_tensor(x): 
  image_string = tf.read_file(x)
  image_decoded = tf.image.decode_image(image_string)
  return image_decoded

'''
正则化图片数据
'''
def image_tensor_norm(image_tensor):
  return tf.divide(tf.subtract(tf.cast(image_tensor, tf.float32), 128.), 128.)


def _read_image(x, y):
    image = read_image_tensor(x)
    image = image_tensor_norm(image)
    return image, y
    
'''
创建训练集的Dataset Iterator
'''
def train_dataset_iterator(x_train, y_train,
    batch=1,
    shuffle=True, shuffle_buffer_size=100):

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    if shuffle:
        train_ds = train_ds.shuffle(shuffle_buffer_size)
    train_ds = train_ds.map(_read_image).batch(batch)
    iterator = train_ds.make_initializable_iterator()
    return iterator

def validate_dataset_iterator(x_test, y_test):
    validate_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(1000).map(_read_image).batch(1000)
    iterator = validate_ds.make_initializable_iterator()
    return iterator