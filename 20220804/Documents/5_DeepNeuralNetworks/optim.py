"""
    采用某种算法来对权重进行更新
    所有算法都带下列参数：
        w：权重
        dw：已经计算出的当前权重的梯度
        config：与该算法相关的超参数设置。所有算法都包含learning_rate超参数
    返回值：
        完成更新后的权重w
        config
"""
import numpy as np

def sgd(w, dw, config=None):
    """
        config包含：
        learning_rate 
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)

    w -= config['learning_rate'] * dw
    return w, config

def sgd_momentum(w, dw, config=None):
    """
    config包含:
        learning_rate
        momentum: 0~1之间的浮点数。当取值为0时，就退化为标准sgd
        velocity: 存放速度v值(矩阵)
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9)
    # 从config中取出之前的循环中计算出的v
    v = config.get('velocity', np.zeros_like(w))
    v = config['momentum'] * v - config['learning_rate'] * dw
    next_w = w + v
    # 将新的v值保存回config中以供后续循环使用
    config['velocity'] = v

    return next_w, config


def rmsprop(w, dw, config=None):
    """
    config包括：
        learning_rate
        decay_rate：0~1之间的浮点数
        epsilon：防止计算溢出的非常小的数
        cache: 存放之前循环已经计算出的累加结果
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('decay_rate', 0.99)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('cache', np.zeros_like(w))

    config['cache'] = config['decay_rate'] * config['cache'] + (1 - config['decay_rate']) * dw**2
    next_w = w - config['learning_rate'] * dw / (np.sqrt(config['cache']) + config['epsilon'])

    return next_w, config


def adam(w, dw, config=None):
    """
    config包括：
        learning_rate
        beta1:
        beta2:
        epsilon:
        m: 
        v:
        t:
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', np.zeros_like(w))
    config.setdefault('v', np.zeros_like(w))
    config.setdefault('t', 1)

    config['t'] += 1
    config['m'] = config['beta1'] * config['m'] + (1 - config['beta1']) * dw
    config['v'] = config['beta2'] * config['v'] + (1 - config['beta2']) * (dw ** 2)

    mt = config['m'] / (1-config['beta1'] ** config['t'])
    vt = config['v'] / (1-config['beta2'] ** config['t'])

    next_w = w - config['learning_rate'] * mt / (np.sqrt(vt) + config['epsilon'])

    return next_w, config