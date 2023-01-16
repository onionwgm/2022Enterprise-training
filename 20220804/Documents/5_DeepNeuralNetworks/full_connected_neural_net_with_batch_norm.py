''' 包含batch norm特性的全连接神经网络 '''
import numpy as np
from util_common_layer import *
from util_batch_norm_layer import *

class FullyConnectedNet(object):
    """
    可支持Batch Normalization的多层全连接神经网络，设计有L个隐藏层
    {affine - [batch-norm] - relu } x L - affine - softmax
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 weight_scale=1e-2, reg=0.0, use_batchnorm=False):
        """
        初始化神经网络系数
        输入参数：
        input_dim： 样本的特征数量
        hidden_dims： 以一维数组的形式，依次指示每个隐藏层的神经元数量
        num_classes： 分类数量
        weight_scale： 权重初始化的缩放因子
        reg：惩罚系数
        use_batchnorm：指示是否使用batch normalization
        """
        self.reg = reg
        self.use_batchnorm = use_batchnorm
        self.num_layers = 1 + len(hidden_dims)      # 隐藏层+输出层
        self.params = {}
     
        modif_hidden_dims = [input_dim] + hidden_dims + [num_classes]

        # 初始化每层的权重和偏置
        for i in range(0, self.num_layers):
            W_name = 'W' + str(i+1)
            b_name = 'b' + str(i+1)
            # 如果启用了batch norm，则需要为每层设置gamma和beta
            if use_batchnorm and i != (self.num_layers-1):
                gamma_name = 'gamma' + str(i+1)
                beta_name = 'beta' + str(i+1)
                self.params[gamma_name] = np.ones(modif_hidden_dims[i+1])
                self.params[beta_name] = np.zeros(modif_hidden_dims[i+1])

            self.params[b_name] = np.zeros(modif_hidden_dims[i+1])
            self.params[W_name] = np.random.normal(scale=weight_scale, size=(modif_hidden_dims[i], modif_hidden_dims[i+1]))
        
        # 先设置每层的batch norm为train模式
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

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

        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        self.cache = {}
        self.batchnorm_cache = {}
        scores = X

        for i in range(1, self.num_layers+1):
            id_str = str(i)
            W_name = 'W' + id_str
            b_name = 'b' + id_str
            gamma_name = 'gamma' + id_str
            beta_name = 'beta' + id_str
            batchnorm_name = 'batchnorm' + id_str
            cache_affine_name = 'c_affine_' + id_str
            cache_relu_name = 'c_relu_' + id_str

            if i == self.num_layers:
                scores, affine_cache = affine_forward(scores, self.params[W_name], self.params[b_name])
            else:
                scores, affine_cache = affine_forward(scores, self.params[W_name], self.params[b_name])
                if self.use_batchnorm:
                    scores, self.batchnorm_cache[batchnorm_name] = batchnorm_forward(scores, self.params[gamma_name], self.params[beta_name], self.bn_params[i-1])
                scores, relu_cache = relu_forward(scores)
               
            self.cache[cache_affine_name] = affine_cache
            self.cache[cache_relu_name] = relu_cache

        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        loss, der = softmax_loss_grad(scores, y)

        for i in range(self.num_layers, 0, -1):
            id_str = str(i)
            W_name = 'W' + id_str
            b_name = 'b' + id_str
            gamma_name = 'gamma' + id_str
            beta_name = 'beta' + id_str
            batchnorm_name = 'batchnorm' + id_str
            cache_affine_name = 'c_affine_' + id_str
            cache_relu_name = 'c_relu_' + id_str

            loss += 0.5*self.reg*np.sum(self.params[W_name]**2) 
            
            if i == self.num_layers:
                der, grads[W_name], grads[b_name] = affine_backward(der, self.cache[cache_affine_name])
            else:
                der = relu_backward(der, self.cache[cache_relu_name])

                if self.use_batchnorm:
                    der, grads[gamma_name], grads[beta_name] = batchnorm_backward(der, self.batchnorm_cache[batchnorm_name])

                der, grads[W_name], grads[b_name] = affine_backward(der, self.cache[cache_affine_name])
            
            grads[W_name] += self.reg*self.params[W_name]
            
        return loss, grads
