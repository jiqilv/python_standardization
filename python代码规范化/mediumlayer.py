from tensorflow.keras.layers import Layer
import tensorflow as tf
from tensorflow.keras import backend as K


class MediumLayer(Layer):
    """
    中间层

    参数:
    - kwargs: 其他关键字参数
    """

    def __init__(self, **kwargs):
        super(MediumLayer, self).__init__(**kwargs)

    def build_layer(self, input_shape):
        """
        构建层

        参数:
        - input_shape: 输入的形状
        """
        super(MediumLayer, self).build(input_shape)

    def compute_output_tensor(self, inputs, **kwargs):
        """
        调用层进行前向传播

        参数:
        - inputs: 输入张量

        返回值:
        - sentence_token_level_outputs: 句子-单词级别的输出张量
        """
        sentence_token_level_outputs = tf.stack(inputs, axis=0)
        sentence_token_level_outputs = K.permute_dimensions(sentence_token_level_outputs, (1, 0, 2))
        return sentence_token_level_outputs

    def compute_output_shape(self, input_shape):
        """
        计算输出形状

        参数:
        - input_shape: 输入的形状

        返回值:
        - output_shape: 输出的形状
        """
        return (input_shape[0][0], len(input_shape), input_shape[0][1])


'''
# 测试代码
x1 = tf.random.truncated_normal([100, 150])
x2 = tf.random.truncated_normal([100, 150])
x3 = tf.random.truncated_normal([100, 150])
x4 = tf.random.truncated_normal([100, 150])

w = MediumLayer()([x1, x2, x3, x4])
print(w)
'''
