import os


class AppConf(object):
    root_path = os.path.split(os.path.realpath(__file__))[0]
    data_path = os.path.join(root_path, 'data')
    pretrained_path = os.path.join(data_path, 'pretrained')
    processed_path = os.path.join(data_path, 'processed')
    raw_path = os.path.join(data_path, 'raw')

    raw_path_en = os.path.join(raw_path, "en")
    raw_path_en_label = os.path.join(raw_path, "en_truth.txt")

    # raw_path_2014_train = os.path.join(raw_path_2014, "Restaurants_Train_2014.xml")
    # raw_path_2014_test = os.path.join(raw_path_2014, "Restaurants_Test_Data_phaseB.xml")



    # yelp_reviews_json_path = os.path.join(raw_path, 'yelp_academic_dataset_review.json')
    # yelp_reviews_txt_path = os.path.join(raw_path, 'yelp_academic_dataset_review.txt')

    processed_path_en = os.path.join(processed_path, 'fakenews_en')
    processed_train_path_en = os.path.join(processed_path, 'train_en')
    processed_test_path_en = os.path.join(processed_path, 'test_en')
    processed_val_path_en = os.path.join(processed_path, 'val_en')


    pretrainned_path_en = os.path.join(pretrained_path, 'en')
    pretrainned_path_2016 = os.path.join(pretrained_path, 'semeval2016')

    train_test_val_dataset_file_2014 = os.path.join(pretrainned_path_en, 'train_test_val_en.pkl')
    train_test_val_dataset_file_2016 = os.path.join(pretrainned_path_2016, 'train_test_val_2016.pkl')

    pretrainned_word_embedding = os.path.join(pretrained_path, 'word_embedding')
    pretrainned_word_embedding_en= os.path.join(pretrainned_word_embedding, 'en')
    pretrainned_word_embedding_2016 = os.path.join(pretrainned_word_embedding, '2016')
    word2id_file_en= os.path.join(pretrainned_word_embedding_en, 'word2id_en.pkl')
    word2id_file_2016 = os.path.join(pretrainned_word_embedding_2016, 'word2id_2016.pkl')

    glove_path = os.path.join(pretrained_path, 'glove')
    word2vec_path = os.path.join(pretrained_path, 'word2vec')
    embedding_path_en = os.path.join(pretrainned_word_embedding_en, 'en')
    # embedding_path_2016 = os.path.join(pretrainned_word_embedding_2016, 'semeval2016')
    max_sen = 50
    sentenceEncoding_r = 30

    output_data_path = os.path.join(data_path, 'output')
    seeds_2014 = {
        'food': ['food', 'taste', 'delicious', 'authentic', 'chicken', 'beef', 'steak', 'pizza', 'lunch', 'cheese',
                 'sauce', 'menu', 'desserts'],
        'price': ['prices', 'dollars', 'cheap', 'inexpensive', 'expensive', 'priced', 'money', 'value', 'bill', 'cost',
                  'overpriced', 'price'],
        'service': ['waiter', 'service', 'friendly', 'staff', 'reservation'],
        'ambience': ['place', 'ambiance', 'atmosphere', 'quiet', 'comfortable', 'noisy']}
