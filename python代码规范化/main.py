from __future__ import print_function
from __future__ import print_function
import os
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
import random
import pickle
import logging
import argparse
import numpy as np
import warnings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")
random.seed(42)
warnings.filterwarnings("ignore")

'''
tf.compat.v1.set_random_seed(1)  # 图级种子，使所有操作会话生成的随机序列在会话中可重复，请设置图级种子：
random.seed(1)  # 让每次生成的随机数一致
np.random.seed(1)  #
'''
set_session = tf.compat.v1.keras.backend.set_session

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.55  # half of the memory
set_session(tf.compat.v1.Session(config=config))

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)


class StandoneCode:
    # dict.get(）：返回指定键的值，如果键不在字典中返回默认值 None 或者设置的默认值
    def __init__(self, conf=None):
        self.conf = dict() if conf is None else conf
        self._buckets = conf.get('buckets', [(2, 10, 22, 72), (2, 20, 34, 102), (2, 40, 34, 202), (2, 100, 34, 302)])
        self._buckets_text_max = (max([i for i, _, _, _ in self._buckets]), max([j for _, j, _, _ in self._buckets]))
        self._buckets_code_max = (max([i for _, _, i, _ in self._buckets]), max([j for _, _, _, j in self._buckets]))
        self.path = self.conf.get('workdir', './data/')
        self.train_params = conf.get('training_params', dict())
        self.data_params = conf.get('data_params', dict())
        self.model_params = conf.get('model_params', dict())
        self._eval_sets = None

    def load_pickle(self, filename):
        with open(filename, 'rb') as f:
            word_dict = pickle.load(f)
        return word_dict

        ##### Data Set #####

    ##### Padding #####
    def pad(self, data, len=None):
        from keras.preprocessing.sequence import pad_sequences
        return pad_sequences(data, maxlen=len, padding='post', truncating='post', value=0)

    ##### Model Loading / saving #####
    def save_model_epoch(self, model, epoch, d12, d3, d4, d5, r):
        if not os.path.exists(self.path + 'models/' + self.model_params['model_name'] + '/'):
            os.makedirs(self.path + 'models/' + self.model_params['model_name'] + '/')
        model.save("{}models/{}/pysparams:d12={}_d3={}_d4={}_d5={}_r={}_epo={:d}_class.h5".format(self.path,
                                                                                                  self.model_params[
                                                                                                      'model_name'],
                                                                                                  d12, d3, d4, d5, r,
                                                                                                  epoch),
                   overwrite=True)

    def load_model_epoch(self, model, epoch, d12, d3, d4, d5, r):
        assert os.path.exists(
            "{}models/{}/pysparams:d12={}_d3={}_d4={}_d5={}_r={}_epo={:d}_class.h5".format(self.path, self.model_params[
                'model_name'], d12, d3, d4, d5, r, epoch)) \
            , "Weights at epoch {:d} not found".format(epoch)
        model.load("{}models/{}/pysparams:d12={}_d3={}_d4={}_d5={}_r={}_epo={:d}_class.h5".format(self.path,
                                                                                                  self.model_params[
                                                                                                      'model_name'],
                                                                                                  d12, d3, d4, d5, r,
                                                                                                  epoch))

    def del_pre_model(self, prepoch, d12, d3, d4, d5, r):
        if (len((prepoch))) >= 2:
            lenth = len(prepoch)
            epoch = prepoch[lenth - 2]
            if os.path.exists("{}models/{}/pysparams:d12={}_d3={}_d4={}_d5={}_r={}_epo={:d}_class.h5".format(self.path,
                                                                                                             self.model_params[
                                                                                                                 'model_name'],
                                                                                                             d12, d3,
                                                                                                             d4, d5, r,
                                                                                                             epoch)):
                os.remove("{}models/{}/pysparams:d12={}_d3={}_d4={}_d5={}_r={}_epo={:d}_class.h5".format(self.path,
                                                                                                         self.model_params[
                                                                                                             'model_name'],
                                                                                                         d12, d3, d4,
                                                                                                         d5, r, epoch))

    def process_instance(self, instance, target, maxlen):
        w = self.pad(instance, maxlen)
        target.append(w)

    def process_matrix(self, inputs, trans1_length, maxlen):
        inputs_trans1 = np.split(inputs, trans1_length, axis=1)
        processed_inputs = []
        for item in inputs_trans1:
            item_trans2 = np.squeeze(item, axis=1).tolist()
            processed_inputs.append(item_trans2)
        return processed_inputs

    def get_data(self, path):
        data = self.load_pickle(path)  # ,self.data_params['train_path']
        text_block_length, text_word_length, query_word_length, code_token_length = 2, 100, 25, 350
        text_blocks = self.process_matrix(np.array([samples_term[1] for samples_term in data]),
                                          text_block_length, 100)

        text_S1 = text_blocks[0]
        text_S2 = text_blocks[1]

        code_blocks = self.process_matrix(np.array([samples_term[2] for samples_term in data]),
                                          text_block_length - 1, 350)
        code = code_blocks[0]

        queries = [samples_term[3] for samples_term in data]
        labels = [samples_term[5] for samples_term in data]
        ids = [samples_term[0] for samples_term in data]

        return text_S1, text_S2, code, queries, labels, ids
        # return text_S1, text_S2, code, queries, ids

    random.seed(42)
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")
    warnings.filterwarnings("ignore")

    tf.compat.v1.set_random_seed(1)
    np.random.seed(1)

    set_session = tf.compat.v1.keras.backend.set_session
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.55
    set_session(tf.compat.v1.Session(config=config))

    seed = 42
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)

    class StandoneCode:
        def __init__(self, conf=None):
            self.conf = dict() if conf is None else conf
            self._buckets = conf.get('buckets',
                                     [(2, 10, 22, 72), (2, 20, 34, 102), (2, 40, 34, 202), (2, 100, 34, 302)])
            self._buckets_text_max = (
                max([i for i, _, _, _ in self._buckets]), max([j for _, j, _, _ in self._buckets]))
            self._buckets_code_max = (
                max([i for _, _, i, _ in self._buckets]), max([j for _, _, _, j in self._buckets]))
            self.path = self.conf.get('workdir', './data/')
            self.train_params = conf.get('training_params', dict())
            self.data_params = conf.get('data_params', dict())
            self.model_params = conf.get('model_params', dict())
            self._eval_sets = None

        def load_pickle(self, filename):
            with open(filename, 'rb') as f:
                word_dict = pickle.load(f)
            return word_dict

        def pad(self, data, len=None):
            return pad_sequences(data, maxlen=len, padding='post', truncating='post', value=0)

        def save_model_epoch(self, model, epoch, d12, d3, d4, d5, r):
            if not os.path.exists(self.path + 'models/' + self.model_params['model_name'] + '/'):
                os.makedirs(self.path + 'models/' + self.model_params['model_name'] + '/')
            model.save("{}models/{}/pysparams:d12={}_d3={}_d4={}_d5={}_r={}_epo={:d}_class.h5".format(self.path,
                                                                                                      self.model_params[
                                                                                                          'model_name'],
                                                                                                      d12, d3, d4, d5,
                                                                                                      r, epoch),
                       overwrite=True)

        def load_model_epoch(self, model, epoch, d12, d3, d4, d5, r):
            assert os.path.exists(
                "{}models/{}/pysparams:d12={}_d3={}_d4={}_d5={}_r={}_epo={:d}_class.h5".format(self.path,
                                                                                               self.model_params[
                                                                                                   'model_name'], d12,
                                                                                               d3, d4, d5, r,
                                                                                               epoch)), "Weights at epoch {:d} not found".format(
                epoch)
            model.load("{}models/{}/pysparams:d12={}_d3={}_d4={}_d5={}_r={}_epo={:d}_class.h5".format(self.path,
                                                                                                      self.model_params[
                                                                                                          'model_name'],
                                                                                                      d12, d3, d4, d5,
                                                                                                      r, epoch))

        def del_pre_model(self, preepoch, d12, d3, d4, d5, r):
            if len(prepoch) >= 2:
                lenth = len(prepoch)
                epoch = prepoch[lenth - 2]
                if os.path.exists(
                        "{}models/{}/pysparams:d12={}_d3={}_d4={}_d5={}_r={}_epo={:d}_class.h5".format(self.path,
                                                                                                       self.model_params[
                                                                                                           'model_name'],
                                                                                                       d12, d3, d4, d5,
                                                                                                       r, epoch)):
                    os.remove("{}models/{}/pysparams:d12={}_d3={}_d4={}_d5={}_r={}_epo={:d}_class.h5".format(self.path,
                                                                                                             self.model_params[
                                                                                                                 'model_name'],
                                                                                                             d12, d3,
                                                                                                             d4, d5, r,
                                                                                                             epoch))

        def process_instance(self, instance, target, maxlen):
            w = self.pad(instance, maxlen)
            target.append(w)

        def process_matrix(self, inputs, trans1_length, maxlen):
            inputs_trans1 = np.split(inputs, trans1_length, axis=1)
            processed_inputs = []
            for item in inputs_trans1:
                item_trans2 = np.squeeze(item, axis=1).tolist()
                processed_inputs.append(item_trans2)
            return processed_inputs

        def get_data(self, path):
            data = self.load_pickle(path)
            text_S1 = []
            text_S2 = []
            code = []
            queries = []
            labels = []
            id = []

            text_block_length, text_word_length, query_word_length, code_token_length = 2, 100, 25, 350
            text_blocks = self.process_matrix(np.array([samples_term[1] for samples_term in data]),
                                              text_block_length, 100)

            text_S1 = text_blocks[0]
            text_S2 = text_blocks[1]

            code_blocks = self.process_matrix(np.array([samples_term[2] for samples_term in data]),
                                              text_block_length - 1, 350)
            code = code_blocks[0]

            queries = [samples_term[3] for samples_term in data]
            labels = [samples_term[5] for samples_term in data]
            ids = [samples_term[0] for samples_term in data]

            return text_S1, text_S2, code, queries, labels, ids


def parse_args():
    parser = argparse.ArgumentParser("Train and Test Model")
    parser.add_argument("--train", choices=["python", "sql"], default="python", help="train dataset set")
    parser.add_argument("--mode", choices=["train", "eval"], default="train",
                        help="The mode to run. The `train` mode trains a model; the `eval` mode evaluates models in a test set")
    parser.add_argument("--verbose", action="store_true", default=True, help="Be verbose")
    return parser.parse_args()


class StandoneCode:
    def __init__(self, config):
        self.data_params = config['data_params']
        self.train_params = config['training_params']

    def get_data(self, path):
        # 从给定路径加载数据的逻辑
        pass

    def save_model_epoch(self, model, epoch, dropout1, dropout2, dropout3, dropout4, regularizer):
        # 保存模型的逻辑
        pass

    def load_model_epoch(self, model, epoch, dropout1, dropout2, dropout3, dropout4, regularizer):
        # 加载模型的逻辑
        pass

    def params_adjust(self, dropout1, dropout2, dropout3, dropout4, dropout5, regularizer, num, seed):
        # 调整模型参数的逻辑
        pass

    def build(self):
        # 构建模型的逻辑
        pass

    def train(self, model):
        # 训练模型的逻辑
        pass

    def valid(self, model, path):
        # 在验证集上评估模型的逻辑
        pass

    def eval(self, model, path):
        # 在测试集上评估模型的逻辑
        pass


if __name__ == '__main__':
    args = parse_args()
    conf = get_config(args.train)
    train_path = conf['data_params']['train_path']
    dev_path = conf['data_params']['valid_path']
    test_path = conf['data_params']['test_path']

    model = eval(conf['model_params']['model_name'])(conf)

    StandoneCode = StandoneCode(conf)

    drop1 = drop2 = drop3 = drop4 = drop5 = 0.8
    r = 0.0002

    conf['training_params']['regularizer'] = 8
    model.params_adjust(dropout1=drop1, dropout2=drop2, dropout3=drop3, dropout4=drop4, dropout5=drop5,
                        Regularizer=round(r, 5), num=8, seed=42)
    conf['training_params']['dropout1'] = drop1
    conf['training_params']['dropout2'] = drop2
    conf['training_params']['dropout3'] = drop3
    conf['training_params']['dropout4'] = drop4
    conf['training_params']['dropout5'] = drop5
    conf['training_params']['regularizer'] = round(r, 5)
    conf['training_params']['num'] = 8
    conf['training_params']['seed'] = 42

    if args.mode == 'train':
        model.build()
        StandoneCode.train(model)
    elif args.mode == 'eval':
        model.load_model_epoch(epoch=21, dropout1=drop1, dropout2=drop2, dropout3=drop3, dropout4=drop4,
                               dropout5=drop5, regularizer=round(r, 5))
        StandoneCode.eval(model, test_path)
