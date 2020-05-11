from conf import AppConf
import os
import pickle
import torch
from data_processing.data_filter import get_all_files


torch.cuda.current_device()
import numpy as np


def padding(sent, sequence_len):
    if len(sent) > sequence_len:
        sent = sent[:sequence_len]
    padding = sequence_len - len(sent)
    sent2idx = sent + [0] * padding
    return sent2idx, len(sent)


def data_2tensor(vocab_path, train_path, test_path, label_path, device):
    with open(vocab_path, 'rb') as f:
        word2id, id2word = pickle.load(f)
        f.close()

    label = {}

    with open(label_path, 'r') as f:
        for r in f.readlines():
            r = r.strip('\n').split(':::')
            label[r[0]] = r[1]
        f.close()

    train_file_list = get_all_files(train_path)
    train_dataset = []
    train_dataset_len = []
    train_dataset_y = []
    for name, document in train_file_list.items(): #name:str, document:list(100sentence)
        doc_sen_id = []
        doc_sen_len = []
        for sen in document:
            sen_id = []
            [sen_id.append(word2id.get(word)) for word in sen.split()]
            sen2id, sen_len = padding(sen_id, AppConf.max_sen)
            doc_sen_id.append(sen2id)
            doc_sen_len.append(sen_len)
        train_dataset.append(doc_sen_id)
        train_dataset_len.append(doc_sen_len)
        train_dataset_y.append(int(label[name]))
    train_dataset_tensor = torch.from_numpy(
        np.array(train_dataset)).type(torch.LongTensor).to(device)
    train_dataset_length_tensor = torch.from_numpy(
        np.array(train_dataset_len)).type(torch.LongTensor).to(device)
    train_dataset_tensor_y = torch.from_numpy(
        np.array(train_dataset_y)).type(torch.FloatTensor).to(device)
    train_dataset_pack = [[data, length, target] for (data, length, target) in 
                        zip(train_dataset_tensor, train_dataset_length_tensor, train_dataset_tensor_y)]
        

    test_file_list = get_all_files(test_path)
    test_dataset = []
    test_dataset_len = []
    test_dataset_y = []
    for name, document in test_file_list.items(): #name:str, document:list(100sentence)
        doc_sen_id = []
        doc_sen_len = []
        for sen in document:
            sen_id = []
            [sen_id.append(word2id.get(word)) for word in sen.split()]
            sen2id, sen_len = padding(sen_id, AppConf.max_sen)
            doc_sen_id.append(sen2id)
            doc_sen_len.append(sen_len)
        test_dataset.append(doc_sen_id)
        test_dataset_len.append(doc_sen_len)
        test_dataset_y.append(int(label[name]))
    test_dataset_tensor = torch.from_numpy(
        np.array(test_dataset)).type(torch.LongTensor)
    test_dataset_length_tensor = torch.from_numpy(
        np.array(test_dataset_len)).type(torch.LongTensor)
    test_dataset_tensor_y = torch.from_numpy(
        np.array(test_dataset_y)).type(torch.LongTensor)
    test_dataset_pack = [[data, length, target] for (data, length, target) in
                        zip(test_dataset_tensor, test_dataset_length_tensor, test_dataset_tensor_y)]


    return train_dataset_pack, test_dataset_pack
