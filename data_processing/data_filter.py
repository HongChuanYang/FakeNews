import xml.dom.minidom as xmldom
from conf import AppConf
import os
import random
import numpy as np
import shutil


from data_processing.seed_init import init_seed

init_seed()



def get_raw(path):
    xml_file = xmldom.parse(path)
    eles = xml_file.documentElement
    return eles


def classify_en_raw(source_path, target_path):
    if (os.path.exists(source_path)):
        files = os.listdir(source_path)  # 得到文件夹下所有文件名称
    for file in files :
        eles = get_raw(os.path.join(source_path,file))
        sentence_list = []
        documents = eles.getElementsByTagName("documents")[0]
        document = documents.getElementsByTagName("document")
        for s in document:
            sentence = s.firstChild.data.strip().lower()
            sentence_list.append(sentence)
        # sentence_path = file.rstrip('.xml')
        sentence_path = file.replace("xml", "txt", 1)
        path = os.path.join(target_path, sentence_path)
        with open(path, encoding='utf-8', mode='w')as f:
            for line in sentence_list:
                f.write('{}\n'.format(line))



def get_all_files(dir):
    lsdir = os.listdir(dir)
    file_list = {}
    for file in lsdir:
        file_path = os.path.join(dir, file)
        reviews = []
        with open(file_path, encoding='utf-8', mode='r') as f:
            for r in f.readlines():
                reviews.append(r.strip())
                file_list[file.replace('.txt', '')] = reviews
    return file_list


def generate_test(raw_path, train_path, test_path):
    # 把train数据集分一部分出来给验证集
    lsdir = os.listdir(raw_path)
    num_split = int(len(lsdir)*0.8)
    train_set = lsdir[0:num_split]
    test_set = lsdir[num_split:]
    for train_data in train_set:
        train_data_path = os.path.join(raw_path, train_data)
        shutil.copy(train_data_path,train_path)
    for test_data in test_set:
        test_data_path = os.path.join(raw_path, test_data)
        shutil.copy(test_data_path,test_path)


if __name__ == '__main__':
    # classify_en_raw(AppConf.raw_path_en, AppConf.processed_path_en)
    generate_test(AppConf.processed_path_en, AppConf.processed_train_path_en,
                               AppConf.processed_test_path_en)
