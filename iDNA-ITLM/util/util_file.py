# ---encoding:utf-8---
# @Author : yuying0711
# @Email : 2539449171@qq.com
# @IDE : PyCharm
# @File : util_file.py


def load_tsv_format_data(filename, skip_head=True):
    sequences = []
    labels = []

    with open(filename, 'r') as file:
        if skip_head:
            next(file)
        for line in file:
            if line[-1] == '\n':
                line = line[:-1]
            list = line.split('\t')
            sequences.append(list[2])
            labels.append(int(list[1]))

    return sequences, labels
