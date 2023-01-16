''' 装载训练数据集和测试数据集'''

import os
import pickle
import numpy as np

def load_cifar10_batch(filename):
    """ 装载单个data_batch文件 """
    with open(filename, 'rb') as file:
        datadict = pickle.load(file, encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        # 转换成10000x3x32x32维度的多维矩阵。3代表3个颜色通道，32代表图片的长和宽都有32个元素
        X = X.reshape(10000, 3, 32, 32).astype("float")
        Y = np.array(Y)
    return X, Y

def load_cifar10(root_folder):
    """ 从磁盘文件读取数据到相应的集合中 """
    xs = []
    ys = []
    for index in range(1, 6):
        filename = os.path.join(root_folder, 'data_batch_%d' % (index, ))
        X, Y = load_cifar10_batch(filename)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    Xte, Yte = load_cifar10_batch(os.path.join(root_folder, 'test_batch'))
    return Xtr, Ytr, Xte, Yte


def load_and_preprocess_cifar10(root_folder, num_training=49000, num_validation=1000, num_test=1000):
    """
    从磁盘文件中读取数据，并且划分成Train、Val和Test数据集
    """
    X_train, y_train, X_test, y_test = load_cifar10(root_folder)
        
    # 构造Train、Val和Test数据集
    mask = list(range(num_training, num_training + num_validation))     # 默认情况下，49000~50000作为Val数据集
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))    # 默认情况下， 0~49000作为Train数据集
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))        # 默认情况下，0~1000作为Test数据集
    X_test = X_test[mask]
    y_test = y_test[mask]

    return {
      'X_train': X_train, 'y_train': y_train,
      'X_val': X_val, 'y_val': y_val,
      'X_test': X_test, 'y_test': y_test,
    }