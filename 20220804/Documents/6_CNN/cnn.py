from builtins import object
import numpy as np

from util_common_layer import *
from util_conv_layer import *
from fast_layers import *

class ThreeLayerConvNet(object):
    """
    包含一个卷积层、一个全连接层和一个输出层的卷积神经网络：
    conv - relu - 2x2 max pool - affine - relu - softmax
    样本数据：(N, C, H, W)
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        输入参数:
        input_dim: 按照(C, H, W)的顺序给定输入数据的维度
        num_filters: 卷积层的卷积核数量
        filter_size: 单个卷积核中的尺寸(HH和WW)
        hidden_dim: 全连接层中的神经元数量
        num_classes: 分类数量
        weight_scale: 初始化权重参数时的缩放因子
        reg: L2 regularization strength
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        # W1, b1：卷积层的权重参数和偏置参数
        # W2, b2：全连接神经网络隐藏层的权重参数和偏置参数
        # W3, b3：输出层(softmax之前)的权重参数和偏置参数
        self.params['W1'] = np.random.normal(scale=weight_scale, size=(num_filters, input_dim[0], filter_size, filter_size))
        W2_row_size = num_filters * input_dim[1]/2 * input_dim[2]/2
        self.params['W2'] = np.random.normal(scale=weight_scale, size=(int(W2_row_size), hidden_dim)) 
        self.params['W3'] = np.random.normal(scale=weight_scale, size=(hidden_dim, num_classes))

        self.params['b1'] = np.zeros(num_filters)
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['b3'] = np.zeros(num_classes)

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        计算卷积神经网络的结果得分、损失值和反向传播梯度
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # 卷积层参数
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # 池化层参数
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        # 前向计算，得出scores(softmax之前)
        out_1, cache_1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        out_2, cache_2 = affine_relu_forward(out_1, W2, b2)
        out_3, cache_3 = affine_forward(out_2, W3, b3)
        scores = out_3

        if y is None:
            return scores

        # 成本和梯度计算
        loss, grads = 0, {}
        loss, dscores = softmax_loss_grad(scores, y)
        loss += sum(0.5*self.reg*np.sum(W_tmp**2) for W_tmp in [W1, W2, W3])

        dx_3, grads['W3'], grads['b3'] = affine_backward(dscores, cache_3)
        dx_2, grads['W2'], grads['b2'] = affine_relu_backward(dx_3, cache_2)
        dx_1, grads['W1'], grads['b1'] = conv_relu_pool_backward(dx_2, cache_1)

        grads['W3'] += self.reg*self.params['W3']
        grads['W2'] += self.reg*self.params['W2']
        grads['W1'] += self.reg*self.params['W1']

        return loss, grads
