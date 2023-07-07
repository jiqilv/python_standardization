# -*- coding: utf-8 -*-
from __future__ import print_function
from tensorflow.keras import backend as kr
from tensorflow.keras.layers import Layer
import tensorflow as tf
import os
import numpy as np
import random
import tensorflow as tf

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)


class PositionEmbedding(Layer):
    """
    位置编码层

    参数:
    - size: 位置编码的长度，必须为偶数，默认为None
    - mode: 编码方式，可以是'sum'或'concat'，默认为'sum'
    """

    def __init__(self, size=None, mode='sum', **kwargs):
        self.size = size
        self.mode = mode
        super(PositionEmbedding, self).__init__(**kwargs)

    def compute_output_embed(self, x):
        """
        调用层进行前向传播

        参数:
        - x: 输入张量

        返回值:
        - position_embed: 位置编码后的张量
        """

        if (self.size is None) or (self.mode == 'sum'):
            self.size = int(x.shape[-1])
            # 生成位置编码矩阵，并将其与输入张量 x 进行相加或拼接，以获得添加了位置信息的输出张量。
        batch_size, seq_len = kr.shape(x)[0], kr.shape(x)[1]
        position_j = 1. / kr.pow(10000., 2 * kr.arange(self.size / 2, dtype='float32') / self.size)
        position_j = kr.expand_dims(position_j, 0)
        position_i = kr.cumsum(kr.ones_like(x[:, :, 0]), 1) - 1
        position_i = kr.expand_dims(position_i, 2)
        position_matrix = kr.dot(position_i, position_j)
        position_matrix_sine = kr.sin(position_matrix)[..., tf.newaxis]
        position_matrix_sine_cos = kr.cos(position_matrix)[..., tf.newaxis]
        position_matrix = kr.concatenate([position_matrix_sine, position_matrix_sine_cos])
        position_matrix = kr.reshape(position_matrix, (batch_size, seq_len, self.size))

        if self.mode == 'sum':
            position_embed = position_matrix + x
        elif self.mode == 'concat':
            position_embed = kr.concatenate([position_matrix, x], 2)

        return position_embed

    def compute_output_shape(self, input_shape):
        """
        计算输出形状

        参数:
        - input_shape: 输入的形状

        返回值:
        - output_shape: 输出的形状
        """

        if self.mode == 'sum':
            return input_shape
        elif self.mode == 'concat':
            return (input_shape[0], input_shape[1], input_shape[2] + self.size)


'''
# 测试代码
query = tf.random.truncated_normal([100, 50, 150])
w = PositionEmbedding(150, 'concat')(query)
print(w.shape)
'''
