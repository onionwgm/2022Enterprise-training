import numpy as np

def affine_forward(x, w, b):
    """
    仿射变换(线性变换+偏置)的前向计算过程
    输入参数：
    x：(N, d_1, d_2, ..., d_k)  N个样本，每个样本可以具有多组维度，每组维度种的特征数量分别为d_1, d_2...
    w：(D, M)
    b: (M,) 偏置项
    返回值：
    output：(N, M) 仿射变换计算结果
    cache: (x, w, b) 将x, w, b缓存起来以便其它层能够使用
    """
    N = x.shape[0]
    # d_1~d_k各组维度的特征数量相乘，以便将所有维度放在一个行向量中
    D = np.prod(x.shape[1:])
    x2 = np.reshape(x, (N, D))      # x与x2共享同一片数据，但是维持各自的维度
    # 执行仿射变换
    out = x2.dot(w) + b
    cache = (x, w, b)               
    return out, cache


def affine_backward(dout, cache):
    """
    仿射变换函数的反向传播梯度计算
    输入参数：
    dout: (N, M) 下游已经计算出的梯度
    cache: 从其它层接收到的缓存数据，包括：
        x: (N, d_1, ... d_k)
        w: (D, M)
        b: (M,)
    返回值：
    dx: (N, d1, ..., d_k)
    dw: (D, M)
    db: (M,)
    """
    x, w, b = cache
    N = x.shape[0]
    D = np.prod(x.shape[1:])
    x2 = np.reshape(x, (N, D))
    
    # 计算各梯度
    dx2 = np.dot(dout, w.T)         # (N, D)
    dw = np.dot(x2.T, dout)         # (D, M)
    db = np.dot(dout.T, np.ones(N)) # (M,)

    # 将x的梯度shape变换回(N, d_1, d_2,....d_k形式)
    dx = np.reshape(dx2, x.shape)
    return dx, dw, db


def relu_forward(x):
    """
    ReLU激活函数的正向计算
    输入参数：
    x：任意shape
    输出参数：
    out: x的计算输出结果
    cache: 缓存x
    """
    out = np.maximum(0, x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    ReLU激活函数的反向传播梯度计算
    输入参数：
    dout: 下游已经计算出的梯度，可以时是任意shape
    cache: x
    返回值：
    dx：计算出的x的梯度
    """
    x = cache
    # 对于ReLU函数，当x>0时，梯度值就是1;当x<=0时，梯度值为0
    # 再乘以下游的dout即可完成链式梯度计算
    dx = np.array(dout, copy = True)
    dx[x < 0] = 0
    return dx

def affine_relu_forward(x, w, b):
    """
    仿射+ReLU激活联合前向计算单元
    输出参数：
    out: 完成了仿射+ReLU激活计算的结果
    cache: 包含仿射函数和ReLU函数单独计算时的缓存数据
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    """
    仿射+ReLU激活联合反向传播梯度计算单元
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db

def svm_loss_grad(x, y):
    """
    SVM线性分类器的成本值及梯度计算
    输入参数：
    - x: (N, C)，其中x[i, j]代表第i个样本的第j个分类的score.
    - y: (N,)，其中y[i]是第i个样本的实际分类号(0 <= y[i] < C)
    返回值：
    loss: 损失值
    dx: 输入数据x的梯度值
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores.reshape(-1, 1) + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N

    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N

    return loss, dx


def softmax_loss_grad(x, y):
    """
    Softmax线性分类器的成本值及梯度计算
    输入参数：
    - x: (N, C)，其中x[i, j]代表第i个样本的第j个分类的score.
    - y: (N,)，其中y[i]是第i个样本的实际分类号(0 <= y[i] < C)
    返回值：
    loss: 损失值
    dx: 输入数据x的梯度值
    """
    shifted_logits = x - np.max(x, axis=1, keepdims = True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims = True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N

    dx = probs
    dx[np.arange(N), y] -= 1
    dx /= N

    return loss, dx
