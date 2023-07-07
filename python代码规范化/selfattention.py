from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
import tensorflow as tf
import numpy as np
import os
import random

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)

class SelfAttention(Layer):
    def __init__(self, r, da, name, **kwargs):
        # 初始化自注意力层
        # r: 自注意力头的数量
        # da: 自注意力中间维度的大小
        # name: 自注意力层的名称
        self.r = r
        self.da = da
        self.scope = name
        super(SelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # 构建自注意力层的权重
        # input_shape: 输入张量的形状
        # 初始化Ws1权重矩阵，形状为(input_shape[2], self.da)
        self.Ws1 = self.add_weight(
            name='Ws1' + self.scope,
            shape=(input_shape[2], self.da),
            initializer='glorot_uniform',
            trainable=True
        )

        # 初始化Ws2权重矩阵，形状为(self.da, self.r)
        self.Ws2 = self.add_weight(
            name='Ws2' + self.scope,
            shape=(self.da, self.r),
            initializer='glorot_uniform',
            trainable=True
        )

    def compute_output_matrix(self, inputs, **kwargs):
        # 自注意力层的前向传播
        # inputs: 输入张量

        # 计算A1，形状为(batch_size, seq_len, self.da)
        A1 = K.dot(inputs, self.Ws1)
        # 对A1应用tanh激活函数
        A1 = tf.tanh(tf.transpose(A1))
        A1 = tf.transpose(A1)

        # 计算注意力权重矩阵A_T，形状为(batch_size, seq_len, self.r)
        A_T = K.softmax(K.dot(A1, self.Ws2))

        # 将注意力权重矩阵A的维度重新排列为(batch_size, self.r, seq_len)
        A = K.permute_dimensions(A_T, (0, 2, 1))

        # 计算自注意力的输出B，形状为(batch_size, self.da, seq_len)
        B = tf.matmul(A, inputs)

        # 创建一个单位矩阵，并重复复制为(batch_size, self.r, self.r)
        tile_eye = tf.tile(tf.eye(self.r), [tf.shape(inputs)[0], 1])
        tile_eye = tf.reshape(tile_eye, [-1, self.r, self.r])

        # 计算AA_T，形状为(batch_size, self.r, self.r)
        AA_T = tf.matmul(A, A_T) - tile_eye

        # 计算P，形状为(batch_size,)
        P = tf.square(tf.norm(AA_T, axis=[-2, -1], ord='fro'))

        # 返回自注意力的输出B和P
        return [B, P]

    def compute_output_shape(self, input_shape):
        # 计算自注意力层的输出形状
        # input_shape: 输入张量的形状
        # 返回输出形状为[(batch_size, self.da, self.r), (batch_size,)]
        return [(input_shape[0], self.da, self.r), (input_shape[0],)]
