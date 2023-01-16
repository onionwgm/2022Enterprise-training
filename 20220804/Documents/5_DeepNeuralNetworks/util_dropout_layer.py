import numpy as np

def dropout_forward(x, dropout_param):
    """
    dropout前向计算
    输入参数:
    x：任意维度的输入数据
    dropout_param: dictionary，包含下列Key
        p: 冻结的节点比例。p越大，被冻结的节点越多
        mode: 'test' or 'train'
    输出参数:
    out: 与x相同维度
    cache: tuple (dropout_param, mask)。mask记录了drop out后各节点被保留或被抛弃的标记
    """
    p, mode = dropout_param['p'], dropout_param['mode']

    mask = None
    out = None

    if mode == 'train':     # 在train场景中，执行dropout操作
        mask = (np.random.rand(*x.shape) > p) / (1 - p)
        out = x * mask
    elif mode == 'test':    # 在test场景中，无需进行额外操作
        out = x

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    dropout反向传播梯度计算
    输入参数：
    dout: 任意维度，下游梯度计算结果
    cache: 从dropout前向计算中获得的(dropout_param, mask).
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    if mode == 'train':
        dx = dout * mask
    elif mode == 'test':
        dx = dout
    return dx