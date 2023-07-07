
   #放进单词
def put_in_word(corpus1, corpus2):
    word_vocab = set()
    corpora = [corpus1, corpus2]
    for corpus in corpora:
        for data in corpus:
            for part in data[1][0], data[1][1], data[2][0], data[3]:
                word_vocab.update(part)
    print(len(word_vocab))
    return word_vocab


    #构建初步词典
def create_word_dict(filepath1, filepath2, save_path):
    with open(filepath1, 'r') as f:
        total_data1 = eval(f.read())
    with open(filepath2, 'r') as f:
        total_data2 = eval(f.read())

    x1 = put_in_word(total_data1, total_data2)
    with open(save_path, "w") as f:
        f.write(str(x1))


    # 检查把所有的单词放进一个词典并且保存
def merge_word_dict(filepath1, filepath2, save_path):
    with open(filepath1, 'r') as f:
        total_data1 = set(eval(f.read()))
    with open(filepath2, 'r') as f:
        total_data2 = eval(f.read())
    word_set = set()
    for data in total_data2:
        for sub_list in data[1:]:
            for word_list in sub_list:
                for word in word_list:
                    if word not in total_data1:
                        word_set.add(word)
    print(len(total_data1))
    print(len(word_set))
    with open(save_path, "w") as f:
        f.write(str(word_set))


if __name__ == "__main__":
    #====================获取staqc的词语集合===============
    python_hnn = '/home/gpu/RenQ/stqac/data_processing/hnn_process/data/python_hnn_data_teacher.txt'
    python_staqc = '/home/gpu/RenQ/stqac/data_processing/hnn_process/data/staqc/python_staqc_data.txt'
    python_word_dict = '/home/gpu/RenQ/stqac/data_processing/hnn_process/data/word_dict/python_word_vocab_dict.txt'

    sql_hnn = '/home/gpu/RenQ/stqac/data_processing/hnn_process/data/sql_hnn_data_teacher.txt'
    sql_staqc = '/home/gpu/RenQ/stqac/data_processing/hnn_process/data/staqc/sql_staqc_data.txt'
    sql_word_dict = '/home/gpu/RenQ/stqac/data_processing/hnn_process/data/word_dict/sql_word_vocab_dict.txt'

     #create_word_dict(python_hnn, python_staqc, python_word_dict)
     #create_word_dict(sql_hnn, sql_staqc, sql_word_dict)

    #====================获取最后大语料的词语集合的词语集合===============
    new_sql_staqc = '../hnn_process/ulabel_data/staqc/sql_staqc_unlabled_data.txt'
    new_sql_large = '../hnn_process/ulabel_data/large_corpus/multiple/sql_large_multiple_unlable.txt'
    large_word_dict_sql = '../hnn_process/ulabel_data/sql_word_dict.txt'
    create_word_dict(large_word_dict_sql, new_sql_large, large_word_dict_sql)

    new_python_staqc = '../hnn_process/ulabel_data/staqc/python_staqc_unlabled_data.txt'
    new_python_large ='../hnn_process/ulabel_data/large_corpus/multiple/python_large_multiple_unlable.txt'
    large_word_dict_python = '../hnn_process/ulabel_data/python_word_dict.txt'
    #merge_word_dict(python_word_dict, new_python_large, large_word_dict_python)
    #create_word_dict(new_python_staqc, new_python_large, final_word_dict_python)