import numpy as np

def batchnorm_forward(x, gamma, beta, bn_param):
    """
    batch normalization前向计算
    假设样本集总样本数为NAll，通过mini-batch拆分成NBatch个子样本集。
    BN层可能需要对所有的子样本集计算。
    但是针对每个子集计算时，并不能只考虑该子集本身，而是应该将之前已经进行过BN计算的子集也考虑进来。
    本函数采用的策略是：
        (1)计算本子集的mean和var：sample_mean, sample_var
        (2)从缓存中取出之前子集统计出的running_mean,running_var
        (3)应用下列公式，将本子集与之前子集的数据融合：
            running_mean = momentum * running_mean + (1 - momentum) * sample_mean
            running_var = momentum * running_var + (1 - momentum) * sample_var 
            式中，momentum是一个0~1之间的系数 
        (4)将running_mean和running_var缓存起来，供下一批mini-batch子集融合使用

    输入参数：
    x：(N, D)
    gamma：(D,)
    beta：(D,)
    bn_param: Dictionary，包含下列key:
        mode: 指定本计算是'train'还是'test'场合。必须提供次参数
        eps: 用于避免除数为0的一个较小的数值
        momentum: Constant for running mean / variance.
        running_mean: Array of shape (D,) giving running mean of features
        running_var Array of shape (D,) giving running variance of features

    返回值：
    - out: (N, D)，计算结果
    - cache：缓存主要数据以供反向传播使用
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N = x.shape[0]
    D = np.prod(x.shape[1:])
    x = np.reshape(x, (N, D))
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':                 # 在训练场合下
        # 先计算出mini-batch样本本身的mean和var，然后与之前缓存的数据融合
        sample_mean = np.mean(x, axis=0)        # 各特征均值
        sample_var = np.var(x, axis=0)          # 各特征方差
        x_hat = (x - sample_mean) / np.sqrt(sample_var + eps)      # Normalize

        out = x_hat * gamma + beta              # 追加缩放(Scale)和偏移(Shift)

        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var

        cache = {}
        cache['sample_mean'] = sample_mean
        cache['sample_var'] = sample_var
        cache['x_hat'] = x_hat
        cache['x'] = x
        cache['gamma'] = gamma
        cache['beta'] = beta
        cache['eps'] = eps 
    elif mode == 'test':                # 在验证/测试场合下
        # 不要计算test样本的mean和var，而是直接采用train样本已经计算好的mean和var
        x_hat = (x - running_mean) / np.sqrt(running_var)
        out = x_hat * gamma + beta
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    batch normalization反向传播梯度计算
    输入参数：
    dout:(N, D)，下游导数
    cache: 
    返回值：
    dx: (N, D)，x的梯度
    dgamma：(D,)，gamma的梯度
    dbeta：(D,)，beta的梯度
    """
    dx, dgamma, dbeta = None, None, None
    m = dout.shape[0]

    dx_hat = dout * cache['gamma'] 
    dsample_var = np.sum(dx_hat * (cache['x']-cache['sample_mean']) * (-0.5) * (cache['sample_var'] + cache['eps'])**(-1.5), axis=0)
    dsample_mean = np.sum(dx_hat * (-1/np.sqrt(cache['sample_var'] + cache['eps'])) , axis=0) + dsample_var * ((np.sum(-2*(cache['x']-cache['sample_mean']))) / m)

    dx = dx_hat * (1/np.sqrt(cache['sample_var'] + cache['eps'])) + \
        dsample_var * (2*(cache['x']-cache['sample_mean'])/m) + \
        dsample_mean/m

    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * cache['x_hat'], axis=0)

    return dx, dgamma, dbeta