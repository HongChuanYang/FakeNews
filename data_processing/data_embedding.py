from conf import AppConf
import pickle
from data_processing.data_filter import get_all_files
import numpy as np


def words_2id(train_path, test_path, target_path):
    train_file_list = get_all_files(train_path)
    test_file_list = get_all_files(test_path)
    char_word_2id = {'<PAD>': 0, '<UNK>': 1}
    total_reviews = []
    for l_train in train_file_list:
        total_reviews += train_file_list[l_train]
    for l_test in test_file_list:
        total_reviews += test_file_list[l_test]
    for l in total_reviews:
        lists = l.split()
        for word in lists:
            if word not in char_word_2id:
                char_word_2id[word] = len(char_word_2id)
    id2word = {char_word_2id[i]: i for i in char_word_2id}
    print("vocabulary size:{}, including train and test file".format(len(char_word_2id)))
    with open(target_path, 'wb') as f:
        pickle.dump((char_word_2id, id2word), f)


def get_embedding(word_embedding_type, word_embedding_corpus, embedding_path, embedding_size):
    if embedding_path is None:
        return None
    else:
        with open(embedding_path + '.{}.{}.{}d.pkl'.format(word_embedding_type, word_embedding_corpus, embedding_size),
                  'rb') as f2:
            embedding_matrix = pickle.load(f2)
        # with open(embedding_path + 'seed.{}.{}.{}d.pkl'.format(word_embedding_type, word_embedding_corpus,
        #                                                        embedding_size),
        #           'rb') as f2:
        #     seed_embedding_matrix = pickle.load(f2)
        return embedding_matrix


def data_embedding(word_embedding_type_path, word_embedding_type, word_embedding_corpus, vocab_path, embedding_path,
                   embedding_size):
    # 把当前所有训练集的句子统计好总词量后，使用glove词向量代替当前词表里面的每一个单词
    with open(vocab_path, 'rb') as f:
        word2id, id2word = pickle.load(f)
        f.close()

    embedding_dir = {}
    if word_embedding_type == "glove" and word_embedding_corpus == "wiki":
        f = open(word_embedding_type_path + '\glove.6B.' + str(embedding_size) + 'd.txt', encoding='utf-8')
    elif word_embedding_type == "glove" and word_embedding_corpus == "yelp":
        f = open(word_embedding_type_path + '\glove.yelp.' + str(embedding_size) + 'd.txt', encoding='utf-8')
    elif word_embedding_type == "word2vec" and word_embedding_corpus == "yelp":
        f = open(word_embedding_type_path + '\word2vec_' + str(embedding_size) + '.txt', encoding='utf-8')
    elif word_embedding_type == "word2vec" and word_embedding_corpus == "wiki":
        f = open(word_embedding_type_path + '\enwiki_20180420_' + str(embedding_size) + 'd.txt', encoding='utf-8')
    for i, line in enumerate(f):
        try:
            values = line.split()
            if len(values) < 10:
                continue
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embedding_dir[word] = coefs
        except Exception as e:
            print(e)
    f.close()
    print('uniform_init...')
    rng = np.random.RandomState(None)
    embedding_matrix = rng.uniform(-0.25, 0.25, size=(len(word2id), embedding_size))
    find_word2vec = 0
    for i, word in enumerate(word2id):
        if word in embedding_dir:
            embedding_vector = embedding_dir.get(word)
            embedding_matrix[i] = embedding_vector
            find_word2vec += 1

    print('目前词表总共有{}个词'.format(len(word2id)))
    print('embeddings总共有{}个词'.format(len(embedding_dir)))
    print('词表embeddings总共有{}个词'.format(find_word2vec))
    print('embeddings的shape为： {}'.format(np.shape(embedding_matrix)))
    with open(embedding_path + '.{}.{}.{}d.pkl'.format(word_embedding_type, word_embedding_corpus, embedding_size),
              'wb') as f2:
        pickle.dump(embedding_matrix, f2)


if __name__ == '__main__':
    # 把所有训练集的单词统计并编号
    # words_2id(AppConf.processed_train_path_en, AppConf.processed_test_path_en, AppConf.word2id_file_en)
    # 词典embedding
    data_embedding(AppConf.glove_path, 'glove', 'yelp', AppConf.word2id_file_en, AppConf.embedding_path_en,
                   300)
