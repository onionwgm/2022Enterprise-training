import numpy as np
from util_common_layer import *
from fast_layers import *

def conv_forward_naive(x, w, b, conv_param):
    """
    卷积前)计算操作的最原始实现

    输入参数：
    x: (N, C, H, W)，待卷积运算的图片数据，有N张图片样本
    w: (F, C, HH, WW)，卷积权重矩阵，每个矩阵有F个filter，每个Filter包含C个颜色通道，宽HH，高WW
    b: (F,)，偏置项
    conv_param: dictionary，包含下列Key:
        'stride': 水平及垂直方向上卷积移动步长
        'pad': 水平及垂直方向上要补齐的元素数
    返回值：
    out: (N, F, H', W')，其中H'和W'分别是：
        H' = 1 + (H + 2 * pad - HH) / stride
        W' = 1 + (W + 2 * pad - WW) / stride
    cache: (x, w, b, conv_param)
    """
    (N, C, H, W) = x.shape
    (F, _, HH, WW) = w.shape
    stride = conv_param['stride']
    pad = conv_param['pad']
    # 根据stride及pad计算水平和垂直方向上的卷积计算结果数量
    H_prime = int(1 + (H + 2 * pad - HH) / stride)
    W_prime = int(1 + (W + 2 * pad - WW) / stride)
    out = np.zeros((N, F, H_prime, W_prime))

    for n in range(N):
        # 在原始图片矩阵上(不包括颜色通道维度)补齐
        x_pad = np.pad(x[n,:,:,:], ((0,0),(pad,pad),(pad,pad)), 'constant')
        for f in range(F):
            for h_prime in range(H_prime):
                for w_prime in range(W_prime):
                    h1 = h_prime * stride           # 水平方向起点索引
                    h2 = h_prime * stride + HH      # 水平方向重点索引
                    w1 = w_prime * stride
                    w2 = w_prime * stride + WW
                    window = x_pad[:, h1:h2, w1:w2] # 选定卷积区域
                    # 计算单个Filter上每个点的卷积
                    out[n, f, h_prime, w_prime] = np.sum(window * w[f,:,:,:]) + b[f]
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    卷积反向传播梯度计算操作的最原始实现
    输入参数：
    dout：下游梯度
    cache：包含从前向计算中传入的缓存参数：(x, w, b, conv_param)
    返回值：
    dx：x的梯度
    dw: w的梯度
    db: 偏置b的梯度
    """
    (x, w, b, conv_param) = cache
    (N, C, H, W) = x.shape
    (F, _, HH, WW) = w.shape
    (_, _, H_prime, W_prime) = dout.shape
    stride = conv_param['stride']
    pad = conv_param['pad']

    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    for n in range(N):
        dx_pad = np.pad(dx[n,:,:,:], ((0,0),(pad,pad),(pad,pad)), 'constant')
        x_pad = np.pad(x[n,:,:,:], ((0,0),(pad,pad),(pad,pad)), 'constant')
        for f in range(F):
            for h_prime in range(H_prime):
                for w_prime in range(W_prime):
                    h1 = h_prime * stride
                    h2 = h_prime * stride + HH
                    w1 = w_prime * stride
                    w2 = w_prime * stride + WW
                    dx_pad[:, h1:h2, w1:w2] += w[f,:,:,:] * dout[n,f,h_prime,w_prime]
                    dw[f,:,:,:] += x_pad[:, h1:h2, w1:w2] * dout[n,f,h_prime,w_prime]
                    db[f] += dout[n,f,h_prime,w_prime]
        dx[n,:,:,:] = dx_pad[:,1:-1,1:-1]
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    Max-Pooling池化层前向计算的最原始实现
    输入参数：
    x: (N, C, H, W)
    pool_param: dictionary，包含
        'pool_height': 池化区块高度
        'pool_width': 池化区块宽度
        'stride': 步长
    返回值：
    out: 池化结果
    cache: (x, pool_param)
    """
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']

    H_out = 1 + (H - pool_height) / stride 
    W_out = 1 + (W - pool_width) / stride 
    out = np.zeros((N, C, int(H_out), int(W_out)))

    for i in range(0, N):
        x_data = x[i]

        xx, yy = -1, -1
        for j in range(0, H-pool_height+1, stride):
            yy += 1
            for k in range(0, W-pool_width+1, stride):
                xx += 1
                x_rf = x_data[:, j:j+pool_height, k:k+pool_width]
                for l in range(0, C):
                    out[i, l, yy, xx] = np.max(x_rf[l])

            xx = -1
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    Max-Pooling池化层反向传播梯度计算的最原始实现
    输入参数：
    dout：下游梯度
    cache：包含从前向计算中传入的缓存参数：(x, pool_param)
    返回值：
    dx：x的梯度
    """
    x, pool_param = cache
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']

    dx = np.zeros((N, C, H, W))
    H_out = 1 + (H - pool_height) / stride
    W_out = 1 + (W - pool_width) / stride

    for i in range(0, N):
        x_data = x[i]
        xx, yy = -1, -1
        for j in range(0, H-pool_height + 1, stride):
            yy += 1
            for k in range(0, W-pool_width + 1, stride):
                xx += 1
                x_rf = x_data[:, j:j+pool_height, k:k+pool_width]
                for l in range(0, C):
                    x_pool = x_rf[l]
                    mask = x_pool == np.max(x_pool)
                    dx[i, l, j:j+pool_height, k:k+pool_width] += dout[i, l, yy, xx] * mask

            xx = -1
    return dx

def conv_relu_forward(x, w, b, conv_param):
    """ 卷积+ReLU前向计算 """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    out, relu_cache = relu_forward(a)
    cache = (conv_cache, relu_cache)
    return out, cache

def conv_relu_backward(dout, cache):
    """ 卷积+ReLU反向传播计算 """
    conv_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db

def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    """ 卷积+ReLU+池化前向计算 """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    s, relu_cache = relu_forward(a)
    out, pool_cache = max_pool_forward_fast(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache


def conv_relu_pool_backward(dout, cache):
    """ 卷积+ReLU+池化反向传播梯度计算 """
    conv_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_fast(dout, pool_cache)
    da = relu_backward(ds, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db
