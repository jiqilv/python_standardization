def get_config(train):
    """
    获取跑模型运行的配置

    参数:
    - train: 训练集名称

    返回值:
    - conf: 配置字典
    """
    conf = {
        'workdir': f'../train_data/new/origin_model/{train}/',  # 模型保存目录
        'buckets': [(2, 10, 22, 72), (2, 20, 34, 102), (2, 40, 34, 202), (2, 100, 34, 302)],  # 数据桶的大小
        'data_params': {
            # 训练数据、验证数据和测试数据的路径
            'train_path': f'../data/new_data_hnn/{train}/hnn_{train}_train_f.pkl',
            'valid_path': f'../data/new_data_hnn/{train}/hnn_{train}_dev_f.pkl',
            'test_path': f'../data/new_data_hnn/{train}/hnn_{train}_test_f.pkl',
            # 代码和文本预训练词向量的路径
            'code_pretrain_emb_path': f'../data/new_data_hnn/{train}/{train}_word_vocab_final.pkl',
            'text_pretrain_emb_path': f'../data/new_data_hnn/{train}/{train}_word_vocab_final.pkl'
        },
        'training_params': {
            'batch_size': 100,  # 批次大小
            'nb_epoch': 150,  # 迭代次数
            'n_eval': 100,  # 评估频率
            'evaluate_all_threshold': {
                'mode': 'all',
                'top1': 0.4,
            },
            'reload': 0,  # 重新加载模型的时期。如果reload=0，则从头开始训练
            'dropout1': 0,  # 第一个dropout层的丢弃率
            'dropout2': 0,  # 第二个dropout层的丢弃率
            'dropout3': 0,  # 第三个dropout层的丢弃率
            'dropout4': 0,  # 第四个dropout层的丢弃率
            'dropout5': 0,  # 第五个dropout层的丢弃率
            'regularizer': 0,  # 正则化参数
        },
        'model_params': {
            'model_name': 'CodeMF',  # 模型名称
        }
    }
    return conf


def get_config_u2l(train):
    """
    获取打标签运行的配置

    参数:
    - train: 训练集名称

    返回值:
    - conf: 配置字典
    """
    conf = {
        'workdir': f'../train_data/new/fianl/code_sa/{train}/',  # 模型保存目录
        'buckets': [(2, 10, 22, 72), (2, 20, 34, 102), (2, 40, 34, 202), (2, 100, 34, 302)],  # 数据桶的大小
        'data_params': {
            # 训练数据、验证数据和测试数据的路径
            'train_path': f'../data/new_data_hnn/{train}/hnn_{train}_train_f.pkl',
            'valid_path': f'../data/new_data_hnn/{train}/hnn_{train}_dev_f.pkl',
            'test_path': f'../data/new_data_hnn/{train}/hnn_{train}_test_f.pkl',
            # 代码和文本预训练词向量的路径
            'code_pretrain_emb_path': f'../data_processing/hnn_process/ulabel_data/large_corpus/python_word_vocab_final.pkl',
            'text_pretrain_emb_path': f'../data_processing/hnn_process/ulabel_data/large_corpus/python_word_vocab_final.pkl'
        },
        'training_params': {
            'batch_size': 100,  # 批次大小
            'nb_epoch': 150,  # 迭代次数
            'n_eval': 100,  # 评估频率
            'evaluate_all_threshold': {
                'mode': 'all',
                'top1': 0.4,
            },
            'reload': 0,  # 重新加载模型的时期。如果reload=0，则从头开始训练
            'dropout1': 0,  # 第一个dropout层的丢弃率
            'dropout2': 0,  # 第二个dropout层的丢弃率
            'dropout3': 0,  # 第三个dropout层的丢弃率
            'dropout4': 0,  # 第四个dropout层的丢弃率
            'dropout5': 0,  # 第五个dropout层的丢弃率
            'regularizer': 0,  # 正则化参数
        },
        'model_params': {
            'model_name': 'CodeMF',  # 模型名称
        }
    }
    return conf
