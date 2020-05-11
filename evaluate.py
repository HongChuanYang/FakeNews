import argparse
import os

import math
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from conf import AppConf
from data_processing.data_2tensor import data_2tensor
from data_processing.seed_init import init_seed
from model.data_iterator import Iterator
from model.model_factory import create_model
import numpy as np
import torch
import random

train_path = AppConf.processed_train_path_en
test_path = AppConf.processed_test_path_en
label_path = AppConf.raw_path_en_label

vocab_path = AppConf.word2id_file_en
embedding_path = AppConf.embedding_path_en

init_seed()

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--att', type=bool, default=False)
parser.add_argument('--model_name', type=str, default='simpleCNN')
parser.add_argument('--word_embedding_dim', type=str, default='300')
parser.add_argument('--word_embedding_type', type=str, default='glove')
parser.add_argument('--word_embedding_corpus', type=str, default='yelp')
parser.add_argument('--repeat', type=int, default=1)
parser.add_argument('--show_badcase', type=bool, default=False)
parser.add_argument('--asp', type=str, default='fake')
parser.add_argument('--data_source', type=str, default='en')

args = parser.parse_args()
asp_list = os.listdir(train_path)
word_embedding_dim = args.word_embedding_dim
word_embedding_type = args.word_embedding_type
word_embedding_corpus = args.word_embedding_corpus
device = args.device
att = args.att
asp = args.asp
model_name = args.model_name
data_source = args.data_source
show_badcase = args.show_badcase
word_embedding_dim = word_embedding_dim.split(',')
for dim in word_embedding_dim:
    dim = int(dim)
    asp_avg_f1 = []
    asp_avg_acc = []
    asp_avg_recall = []
    asp_avg_presicion = []
    train_dataset_pack, test_dataset_pack = data_2tensor(
        vocab_path, train_path, test_path, label_path, device)
    word2id, id2word, embedding_matrix, model, model_path = create_model(model_name,
                                                                            device, att,
                                                                            vocab_path,
                                                                            embedding_path,
                                                                            data_source, asp,
                                                                            dim,
                                                                            word_embedding_type,
                                                                            word_embedding_corpus)
    # 开始测试
    np.random.shuffle(test_dataset_pack)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    pack = test_dataset_pack
    x_test, l_test, y_test = [i for i in zip(*pack)]
    x_test = torch.stack(x_test).to(device)
    l_test = torch.stack(l_test).to(device)
    y_test = torch.stack(y_test).to(device)
    y_predcit = model.predict(x_test, l_test)
    avg_acc = accuracy_score(y_test.cpu(), y_predcit.cpu())
    avg_f1 = f1_score(y_test.cpu(), y_predcit.cpu())
    avg_precision = precision_score(y_test.cpu(), y_predcit.cpu())
    avg_recall = recall_score(y_test.cpu(), y_predcit.cpu())
    print('======{}:f1:{},acc:{},precision:{},recall:{}'.format(asp, avg_f1, avg_acc, avg_precision,
                                                                avg_recall))
    asp_avg_acc.append(avg_acc)
    asp_avg_f1.append(avg_f1)
    asp_avg_presicion.append(avg_precision)
    asp_avg_recall.append(avg_recall)
    if show_badcase:
        bad_cases = [pack[index] for index, i in enumerate(y_predcit.cpu() == y_test.cpu()) if i == 0]
        for badcase in bad_cases:
            review, l, target = badcase
            review_string = ' '.join([id2word[i.item()] for i in review[0:l]])
            print('{}\t【实际标签：{}】'.format(review_string, target))
    print('model:{},wordembedding:{},dim:{}平均f1:{},acc{},precision:{},recall:{}'.format(model_name, word_embedding_type,
                                                                                        dim,
                                                                                        np.mean(asp_avg_f1),
                                                                                        np.mean(asp_avg_acc),
                                                                                        np.mean(asp_avg_presicion),
                                                                                        np.mean(asp_avg_recall)))
