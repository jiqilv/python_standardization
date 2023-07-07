# 解析结构
from python_structured import *
from sqlang_structured import *
# 多进程
from multiprocessing import Pool as ThreadPool
#python解析
from multiprocessing.pool import ThreadPool
import pickle


    # 选择不同的模型返回结果
def process_python(data_list, mode):
    result = []
    if mode == 'query':
        result = [python_query_parse(line) for line in data_list]
    elif mode == 'code':
        result = [python_code_parse(line) for line in data_list]
    else:
        for line in data_list:
            if line == '-10000':
                result.append(['-10000'])
            else:
                result.append(python_context_parse(line))
    return result


    # 选择不同的模型返回结果
def process_SQL(data_list, mode):
    result = []
    if mode == 'query':
        result = [sqlang_query_parse(line) for line in data_list]
    elif mode == 'code':
        result = [sqlang_query_parse (line)for line in data_list]
    else:
        for line in data_list:
            if line == '-10000':
                result.append(['-10000'])
            else:
                result.append(sqlang_context_parse(line))
    return result

def python_analysis(python_list, split_num):
    acont1_data = [i[1][0][0] for i in python_list]
    acont1_split_list = [acont1_data[i:i + split_num] for i in range(0, len(acont1_data), split_num)]
    pool = ThreadPool(10)
    acont1_list = pool.map(process_python(), acont1_split_list)
    pool.close()
    pool.join()
    acont1_cut = []
    for p in acont1_list:
        acont1_cut += p
    print('acont1条数：%d' % len(acont1_cut))
    acont2_data = [i[1][1][0] for i in python_list]
    acont2_split_list = [acont2_data[i:i + split_num] for i in range(0, len(acont2_data), split_num)]
    pool = ThreadPool(10)
    acont2_list = pool.map(process_python(), acont2_split_list)
    pool.close()
    pool.join()
    acont2_cut = []
    for p in acont2_list:
        acont2_cut += p
    print('acont2条数：%d' % len(acont2_cut))
    query_data = [i[3][0] for i in python_list]
    query_split_list = [query_data[i:i + split_num] for i in range(0, len(query_data), split_num)]
    pool = ThreadPool(10)
    query_list = pool.map(process_python(), query_split_list)
    pool.close()
    pool.join()
    query_cut = []
    for p in query_list:
        query_cut += p
    print('query条数：%d' % len(query_cut))
    code_data = [i[2][0][0] for i in python_list]
    code_split_list = [code_data[i:i + split_num] for i in range(0, len(code_data), split_num)]
    pool = ThreadPool(10)
    code_list = pool.map(process_python(), code_split_list)
    pool.close()
    pool.join()
    code_cut = []
    for p in code_list:
        code_cut += p
    print('code条数：%d' % len(code_cut))
    qids = [i[0] for i in python_list]
    print(qids[0])
    print(len(qids))
    return acont1_cut, acont2_cut, query_cut, code_cut, qids


def sql_analysis(sqlang_list, split_num):
    acont1_data = [i[1][0][0] for i in sqlang_list]
    acont1_split_list = [acont1_data[i:i + split_num] for i in range(0, len(acont1_data), split_num)]
    pool = ThreadPool(10)
    acont1_list = pool.map(process_SQL(), acont1_split_list)
    pool.close()
    pool.join()
    acont1_cut = []
    for p in acont1_list:
        acont1_cut += p
    print('acont1条数：%d' % len(acont1_cut))
    acont2_data = [i[1][1][0] for i in sqlang_list]
    acont2_split_list = [acont2_data[i:i + split_num] for i in range(0, len(acont2_data), split_num)]
    pool = ThreadPool(10)
    acont2_list = pool.map(process_SQL(), acont2_split_list)
    pool.close()
    pool.join()
    acont2_cut = []
    for p in acont2_list:
        acont2_cut += p
    print('acont2条数：%d' % len(acont2_cut))
    query_data = [i[3][0] for i in sqlang_list]
    query_split_list = [query_data[i:i + split_num] for i in range(0, len(query_data), split_num)]
    pool = ThreadPool(10)
    query_list = pool.map(process_SQL(), query_split_list)
    pool.close()
    pool.join()
    query_cut = []
    for p in query_list:
        query_cut += p
    print('query条数：%d' % len(query_cut))
    code_data = [i[2][0][0] for i in sqlang_list]
    code_split_list = [code_data[i:i + split_num] for i in range(0, len(code_data), split_num)]
    pool = ThreadPool(10)
    code_list = pool.map(process_SQL(), code_split_list)
    pool.close()
    pool.join()
    code_cut = []
    for p in code_list:
        code_cut += p
    print('code条数：%d' % len(code_cut))
    qids = [i[0] for i in sqlang_list]
    return acont1_cut, acont2_cut, query_cut, code_cut, qids


def main(lang_type, split_num, source_path, save_path):
    total_data = []
    with open(source_path, "rb") as f:
        corpus_lis = pickle.load(f)
        if lang_type == 'python':
            parse_acont1, parse_acont2, parse_query, parse_code, qids = python_analysis(corpus_lis, split_num)
            for i in range(len(qids)):
                total_data.append([qids[i], [parse_acont1[i], parse_acont2[i]], [parse_code[i]], parse_query[i]])
        if lang_type == 'sql':
            parse_acont1, parse_acont2, parse_query, parse_code, qids = sql_analysis(corpus_lis, split_num)
            for i in range(len(qids)):
                total_data.append([qids[i], [parse_acont1[i], parse_acont2[i]], [parse_code[i]], parse_query[i]])
    with open(save_path, "w") as f:
        f.write(str(total_data))
python_type = 'python'
sqlang_type = 'sql'
words_top = 100
split_num = 1000


def read_file(path1, path2):
    with open(path1, "rb") as f:
        corpus_lis1 = pickle.load(f)
    with open(path2, "rb") as f:
        corpus_lis2 = eval(f.read())
    print(corpus_lis1[10])
    print(corpus_lis2[10])



if __name__ == '__main__':
    staqc_python_path = '../hnn_process/ulabel_data/python_staqc_qid2index_blocks_unlabeled.txt'
    staqc_python_save ='../hnn_process/ulabel_data/staqc/python_staqc_unlabled_data.txt'
    staqc_sql_path = '../hnn_process/ulabel_data/sql_staqc_qid2index_blocks_unlabeled.txt'
    staqc_sql_save = '../hnn_process/ulabel_data/staqc/sql_staqc_unlabled_data.txt'
    #main(sqlang_type,split_num,staqc_sql_path,staqc_sql_save)
    #main(python_type, split_num, staqc_python_path, staqc_python_save)
    large_python_path='../hnn_process/ulabel_data/large_corpus/multiple/python_large_multiple.pickle'
    large_python_save='../hnn_process/ulabel_data/large_corpus/multiple/python_large_multiple_unlable.txt'
    large_sql_path='../hnn_process/ulabel_data/large_corpus/multiple/sql_large_multiple.pickle'
    large_sql_save='../hnn_process/ulabel_data/large_corpus/multiple/sql_large_multiple_unlable.txt'
    #main(sqlang_type, split_num, large_sql_path, large_sql_save)
    main(python_type, split_num, large_python_path, large_python_save)