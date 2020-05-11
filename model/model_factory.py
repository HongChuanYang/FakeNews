from conf import AppConf
from data_processing.data_embedding import get_embedding
# from model.bi_lstm import BiLSTM
# can't install pytorchnlp
from model.word_sentence_encoding_noregularization import WordSentenceEncodingNoregularization
from model.sentences_encoding import SentenceEncoding
from model.simple_cnn import SimpleCNN
# from model.srnn import SRNN
# from model.text_cnn import TextCNN
import pickle
import os

from model.word_sentences_encoding import WordSentenceEncoding


def create_model(model_name, device, att, vocab_path, embedding_path, data_source, classication, embedding_dim,
                 word_embedding_type, word_embedding_corpus):
    with open(vocab_path, 'rb') as f:
        word2id, id2word = pickle.load(f)
        f.close()
    embedding_matrix = get_embedding(word_embedding_type, word_embedding_corpus, embedding_path,
                                     embedding_dim)
    if model_name == 'simpleCNN':
        model = SimpleCNN(len(word2id), 1, embedding_dim, embedding_matrix, device)
        model_path = os.path.join(AppConf.output_data_path,
                                  'simpleCNN_{}_{}_{}_{}_{}.pkl'.format(word_embedding_type, word_embedding_corpus,
                                                                        embedding_dim, data_source, classication
                                                                        ))
    elif model_name == 'textCNN':
        model = TextCNN(len(word2id), 1, embedding_dim, embedding_matrix, device)
        model_path = os.path.join(AppConf.output_data_path,
                                  'textCNN_{}_{}_{}_{}_{}.pkl'.format(word_embedding_type, word_embedding_corpus,
                                                                      embedding_dim, data_source, classication
                                                                      ))
    elif model_name == 'bilstm':
        model = BiLSTM(len(word2id), 1, embedding_dim, embedding_matrix, device, att)
        model_path = os.path.join(AppConf.output_data_path,
                                  'bilstm_{}_{}_{}_{}_{}.pkl'.format(word_embedding_type, word_embedding_corpus,
                                                                     embedding_dim, data_source, classication
                                                                     ))
        if att:
            model_path = os.path.join(AppConf.output_data_path,
                                      'bilstm_{}_{}_{}_{}_{}_att.pkl'.format(word_embedding_type,
                                                                             word_embedding_corpus,
                                                                             embedding_dim, data_source,
                                                                             classication))
    elif model_name == 'srnn':
        seeds = []
        [seeds.extend(items) for (key, items) in AppConf.seeds_2014.items()]
        # if classication != 'anecdotes':
        #     seeds = AppConf.seeds_2014.get(classication)
        # else:
        #     seeds = []
        #     [seeds.extend(items) for (key, items) in AppConf.seeds_2014.items()]
        seeds_ids = [word2id[i] for i in seeds]
        model = SRNN(len(word2id), 1, embedding_dim, embedding_matrix, device, seeds_ids, att)
        model_path = os.path.join(AppConf.output_data_path,
                                  'srnn_{}_{}_{}_{}_{}.pkl'.format(word_embedding_type, word_embedding_corpus,
                                                                   embedding_dim, data_source, classication
                                                                   ))
    elif model_name == 'sentenceEncoding':
        model = SentenceEncoding(len(word2id), 1, embedding_dim, embedding_matrix, device)
        model_path = os.path.join(AppConf.output_data_path,
                                  'sentence_{}_{}_{}_{}_{}.pkl'.format(word_embedding_type, word_embedding_corpus,
                                                                       embedding_dim, data_source, classication
                                                                       ))
    elif model_name == 'word_sentenceEncoding_noregularization':
        model = WordSentenceEncodingNoregularization(len(word2id), 1, embedding_dim, embedding_matrix, device)
        model_path = os.path.join(AppConf.output_data_path,
                                  'word_sentenceEncoding_noregularization_{}_{}_{}_{}_{}.pkl'.format(
                                      word_embedding_type, word_embedding_corpus,
                                      embedding_dim, data_source, classication
                                  ))
    elif model_name == 'word_sentenceEncoding':
        model = WordSentenceEncoding(len(word2id), 1, embedding_dim, embedding_matrix, device)
        model_path = os.path.join(AppConf.output_data_path,
                                  'word_sentence_{}_{}_{}_{}_{}.pkl'.format(word_embedding_type, word_embedding_corpus,
                                                                            embedding_dim, data_source, classication
                                                                            ))

    return word2id, id2word, embedding_matrix, model, model_path
