import os
import random
import numpy as np
import tensorflow as tf
from keras.layers import Layer
from keras import backend as K

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)


class ConcatLayer(Layer):
    def __init__(self, **kwargs):
        super(ConcatLayer, self).__init__(**kwargs)

    def calculate_weight(self, inputs, **kwargs):
        # 将输入张量按照第二个维度进行分割
        block_level_code_output = tf.split(inputs, inputs.shape[1], axis=1)
        # 将分割后的张量在第三个维度上进行连接
        block_level_code_output = tf.concat(block_level_code_output, axis=2)
        # 将维度为1的维度进行压缩
        # 最终输出维度为 (batch_size, 600)
        block_level_code_output = tf.squeeze(block_level_code_output, axis=1)
        print(block_level_code_output)
        return block_level_code_output

    def compute_output_shape(self, input_shape):
        # 输出形状为 (batch_size, 输入维度的乘积)
        print("===========================", input_shape)
        return (input_shape[0], input_shape[1] * input_shape[2])
