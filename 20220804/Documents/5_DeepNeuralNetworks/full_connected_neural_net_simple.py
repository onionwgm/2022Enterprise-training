''' 不包含batch norm, drop out特性的简单全链接神经网络 '''

import numpy as np
from util_common_layer import *

class FullyConnectedNet(object):
    """
    简单的多层全连接神经网络，设计有L个隐藏层
    {affine - relu } x L - affine - softmax
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 weight_scale=1e-2, reg=0.0):
        """
        初始化神经网络系数
        输入参数：
        input_dim： 样本的特征数量
        hidden_dims： 以一维数组的形式，依次指示每个隐藏层的神经元数量
        num_classes： 分类数量
        weight_scale： 权重初始化的缩放因子
        reg：惩罚系数
        """
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)      # 隐藏层+输出层
        self.params = {}
     
        modif_hidden_dims = [input_dim] + hidden_dims + [num_classes]

        # 初始化每层的权重和偏置
        for i in range(0, self.num_layers):
            W_name = 'W' + str(i+1)
            b_name = 'b' + str(i+1)
            self.params[b_name] = np.zeros(modif_hidden_dims[i+1])
            self.params[W_name] = np.random.normal(scale=weight_scale, size=(modif_hidden_dims[i], modif_hidden_dims[i+1]))

    def loss(self, X, y=None):
        """
        X: (N, d_1, ..., d_k)
        y: (N,)，y[i]代表X中第i个样本对应的分类
        返回值：
        如果y是None，则直接计算X中每个样本的scores(N, C)
        如果y不为None，则返回下列内容：
        loss: 成本值
        grads: 各级W和b的梯度
        """
        mode = 'test' if y is None else 'train'

        self.cache = {}
        scores = X

        for i in range(1, self.num_layers+1):
            id_str = str(i)
            W_name = 'W' + id_str
            b_name = 'b' + id_str
            cache_name = 'c' + id_str

            if i == self.num_layers:
                scores, cache = affine_forward(scores, self.params[W_name], self.params[b_name])
            else:
                scores, cache = affine_relu_forward(scores, self.params[W_name], self.params[b_name])

            self.cache[cache_name] = cache

        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        loss, der = softmax_loss_grad(scores, y)

        for i in range(self.num_layers, 0, -1):
            id_str = str(i)
            W_name = 'W' + id_str
            b_name = 'b' + id_str
            cache_name = 'c' + id_str

            loss += 0.5*self.reg*np.sum(self.params[W_name]**2) 
            
            if i == self.num_layers:
                der, grads[W_name], grads[b_name] = affine_backward(der, self.cache[cache_name])
            else:
                der, grads[W_name], grads[b_name] = affine_relu_backward(der, self.cache[cache_name])
            
            grads[W_name] += self.reg*self.params[W_name]
            
        return loss, grads
