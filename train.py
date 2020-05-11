from data_processing.seed_init import init_seed
from model.data_iterator import Iterator
from data_processing.data_2tensor import data_2tensor
from conf import AppConf
from model.model_factory import create_model
import torch
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
import os
import argparse
import numpy as np
import random

init_seed()
train_path = AppConf.processed_train_path_en
test_path = AppConf.processed_test_path_en
label_path = AppConf.raw_path_en_label

vocab_path = AppConf.word2id_file_en
embedding_path = AppConf.embedding_path_en

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--batch_size', type=int, default=8, )
parser.add_argument('--att', type=bool, default=False)
parser.add_argument('--model_name', type=str, default='simpleCNN')
parser.add_argument('--word_embedding_dim', type=int, default=300)
parser.add_argument('--word_embedding_type', type=str, default='glove')
parser.add_argument('--word_embedding_corpus', type=str, default='yelp')
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--epoch', type=int, default=30)
parser.add_argument('--weight_decay', type=float, default=0.00)
parser.add_argument('--data_source', type=str, default='en')
parser.add_argument('--asp', type=str, default='fake')
args = parser.parse_args()
print(args)
word_embedding_dim = args.word_embedding_dim
word_embedding_type = args.word_embedding_type
word_embedding_corpus = args.word_embedding_corpus
device = args.device.strip()
att = args.att
model_name = args.model_name.strip()
batch_size = args.batch_size
epochs = args.epoch
learning_rate = args.learning_rate
weight_decay = args.weight_decay
data_source = args.data_source
asp = args.asp
dim = args.word_embedding_dim
best_f1 = 0

train_dataset_pack, test_dataset_pack = data_2tensor(
    vocab_path, train_path, test_path, label_path, 'cuda:0')

    
np.random.shuffle(train_dataset_pack)
length = len(train_dataset_pack)
train_length = int(length * 0.8)
val_length = length - train_length
print(
    '{}维度,开始{},训练集{}条，验证集{}条================================'.format(dim, asp, train_length,
                                                                     length - train_length))
train_dataset_iterator = Iterator(train_dataset_pack[:train_length])
val_dataset_pack = train_dataset_pack[train_length:]
word2id, id2word, embedding_matrix, model, model_path = create_model(model_name,
                                                                     device, att,
                                                                     vocab_path,
                                                                     embedding_path,
                                                                     data_source, asp,
                                                                     dim,
                                                                     word_embedding_type,
                                                                     word_embedding_corpus)

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for epoch in range(epochs):
    loss_total = 0
    batch_count = 0
    print(
        'epoch:{}==================================================={}模型,device:{},batch_size:{},learning_rate:{}'.format(
            epoch, model_name, device,
            batch_size,
            learning_rate))
    for train_batch in train_dataset_iterator.next(batch_size, shuffle=False):
        pack = train_batch
        x, l, y = [i for i in zip(*pack)]
        x = torch.stack(x)
        l = torch.stack(l)
        y = torch.stack(y)
        model.zero_grad()
        if model_name == 'sentenceEncoding' or model_name == 'word_sentenceEncoding':
            logit, A = model.forward(x, l)
            loss = model.loss(logit.squeeze(-1), y, A)
        else:
            logit = model.forward(x, l)
            loss = model.loss(logit.squeeze(-1), y)
        loss_total += loss.item()
        loss.backward()
        optimizer.step()
        batch_count += 1
        if batch_count % 5 == 0:
            model.eval()
            acc_total = 0
            f1_total = 0
            recall_total = 0
            precision_total = 0
            with torch.no_grad():
                pack = val_dataset_pack
                x_test, l_test, y_test = [i for i in zip(*pack)]
                x_test = torch.stack(x_test).to(device)
                l_test = torch.stack(l_test).to(device)
                y_test = torch.stack(y_test)
                try:
                    y_predcit = model.predict(x_test, l_test)
                except RuntimeError:
                    print('数据出错')
                avg_acc = accuracy_score(y_test.cpu(), y_predcit.cpu())
                avg_f1 = f1_score(y_test.cpu(), y_predcit.cpu(), average='binary')
                avg_precision = precision_score(y_test.cpu(), y_predcit.cpu(), average='binary')
                avg_recall = recall_score(y_test.cpu(), y_predcit.cpu(), average='binary')
            model.train()
            avg_loss = float(loss_total / batch_count)
            if avg_f1 > best_f1:
                best_f1 = avg_f1
                torch.save(model.state_dict(), model_path)
            print(asp + '训练结果：'
                        'acc:{},f1:{},precision:{},recall:{},loss:{},best_f1:{}'.format(avg_acc, avg_f1,
                                                                                        avg_precision,
                                                                                        avg_recall,
                                                                                        avg_loss,
                                                                                        best_f1))
