from __future__ import print_function
from tensorflow.keras.layers import Layer
import os
import numpy as np
import random
import tensorflow as tf

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)


class LayerNormalization(Layer):
    """
    层归一化层
    参数:
    - epsilon: 防止除零的小量
    属性:
    - beta: 偏置项
    - gamma: 缩放因子
    """

    def __init__(self, epsilon=1e-8, **kwargs):
        self._epsilon = epsilon
        super(LayerNormalization, self).__init__(**kwargs)

    def build_model(self, input_shape):
        """
        构建层
        参数:
        - input_shape: 输入的形状
        """
        self.beta = self.add_weight(
            shape=(input_shape[-1],),
            initializer='zero',
            name='beta')
        self.gamma = self.add_weight(
            shape=(input_shape[-1],),
            initializer='one',
            name='gamma')
        super(LayerNormalization, self).build(input_shape)

    def calculate_model(self, inputs):
        """
        调用层进行前向传播
        参数:
        - inputs: 输入张量
        返回值:
        - outputs: 输出张量
        """
        mean, variance = tf.nn.moments(inputs, [-1], keepdims=True)
        normalized = (inputs - mean) / ((variance + self._epsilon) ** 0.5)
        outputs = self.gamma * normalized + self.beta
        return outputs

    def compute_output_shape(self, input_shape):
        """
        计算输出形状
        参数:
        - input_shape: 输入的形状
        返回值:
        - output_shape: 输出的形状
        """
        return input_shape
