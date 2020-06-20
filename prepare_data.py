import string
import json
import pandas as pd
import numpy as np
import cv2
from numpy import array, argmax
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import gensim
import tensorflow as tf

arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
english_punctuations = string.punctuation
punctuations_list = arabic_punctuations + english_punctuations

word2vec_model = gensim.models.Word2Vec.load('ar_wiki_word2vec')


def load_data(file_name, type_):
    '''
    Description: read data from json data, prepare data to fed into model

    Arguments:
        file_name: dataset file name
    Returns:
        X: Features data
        y: Target Class
    '''

    # load data from json file
    data = json.load(open(file_name, encoding='windows-1256'))

    answer_vec = []
    question_index_vec = []
    question_vec = []
    image_id_vec = []

    for i in data['questions']:
        image_id_vec.append(i['img_id'])
        answer_vec.append(i['nla'])
        question_index_vec.append(i['nlq_idx'])  # no need for this line
        question_vec.append(i['nlq'])

    # prepare images
    if (type_ == 'train'):
        image_paths = [get_image_path_train(i) for i in image_id_vec]
    else:
        image_paths = [get_image_path_test(i) for i in image_id_vec]

    iamges = [image_preprocessing(i) for i in image_paths]
    iamges = np.array(iamges)
    print('Done images pre-processing...')

    # prepare questions
    questions_tokenization = tokenization(question_vec)
    questions = [question_preprocessing(i) for i in questions_tokenization]
    questions = np.array(questions)
    questions = questions.reshape(questions.shape[0], 1, questions.shape[1])
    print('Done questions pre-processing...')

    # integrate images and questions in one list
    X = [iamges, questions]

    # encode lables
    y = encode_labels(answer_vec)

    return X, y


def remove_punctuations(text):
    '''
    Description: remove punctuations from text

    Arguments:
        text: string
    Returns:
       text without any punctuations
    '''

    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)


def encode_labels(answers):
    '''
    Description: encode labels

    Arguments:
        answers: array of string (answers)
    Returns:
       encoded: numeric list represent answers after encoded
    '''

    values = array(answers)
    values = array(values)

    label_encoder_train = LabelEncoder( )
    integer_encoded_train = label_encoder_train.fit_transform(values)

    # one hot encode
    encoded = to_categorical(integer_encoded_train)
    return encoded


def get_vector(n_model, dim, token):
    '''
    Description: convert token to vector used pre-trained Word2Vec model

    Arguments:
        n_model: pre-trained Word2Vec model
        dim: vector size
        token: one word
    Returns:
       vec: numeric vector represent one word
    '''
    vec = np.zeros(dim)  # initialization a list with zeros
    if token not in n_model.wv:
        _count = 0
        is_vec = True
        for w in token.split(" "):
            # search for the token in word2vec model
            if w in n_model.wv:
                _count += 1
                vec += n_model.wv[w]
        if _count > 0:
            vec = vec / _count
    else:
        vec = n_model.wv[token]
    return vec


def tokenization(questions):
    '''
    Description: convert string of questions to tokens

    Arguments:
        questions: list of questions
    Returns:
       ##max_len: the length of the maximum question
       questions_list: list of tokenized questions
    '''

    questions_list = []
    questions = [remove_punctuations(question) for question in questions]  # remove punctuations
    questions_list.append([x.split(' ')
                           for x in questions])  # convert question to tokens
    questions_list = questions_list[0]  # remove the duplicate brackets
    # max_len= len(max(questions_list, key=len))  # find the length of the maximum question in the list

    return questions_list


def question_preprocessing(question):
    '''
    Description: processing question (merge tokens vector -  vector padding)

    Arguments:
        question: list of tokens
    Returns:
       padded_inputs: a vector after done pre-processing
    '''

    token_vector = []  # list to store tokens vector
    for i in range(len(question)):
        # get word vector using pre-trained model
        token_vector.append(get_vector(word2vec_model, 300, question[i]))

    token_vector = np.reshape(token_vector, (1, len(token_vector) * 300))  # reshape array to 1 and token vector lenght
    padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(token_vector, dtype='float32',
                                                                  padding='post', maxlen=3000)  # paadding vector
    padded_inputs = padded_inputs.ravel( )  # remove the double brakets
    padded_inputs = np.array(padded_inputs)

    return padded_inputs


def get_image_path_train(image_id):
    '''
    Description: get the image path for training image

    Arguments:
        image_id: integer number
    Returns:
       image_path: string - the image path
    '''

    path = '/Users/afnanbq/PycharmProjects/AVQA/Train_Images'
    if (image_id.endswith('_') or image_id.endswith('#') or image_id.endswith('%')):
        image_path = path + '/COCO_train2014_' + image_id.rjust(13, '0') + '.jpg'
    else:
        image_path = path + '/COCO_train2014_' + image_id.rjust(12, '0') + '.jpg'
    return image_path


def get_image_path_test(image_id):
    '''
    Description: get the image path for testing image

    Arguments:
        image_id: integer number
    Returns:
       image_path: string - the image path
    '''

    path = '/Users/afnanbq/PycharmProjects/AVQA/Test'
    image_path = path + '/COCO_test2015_' + image_id.rjust(12, '0') + '.jpg'
    return image_path


def image_preprocessing(image_path):
    '''
    Description: processing image (resize, normalize)

    Arguments:
        image_path: string - the image path
    Returns:
       image: a vector after done pre-processing
    '''

    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32) / 255  # Normlaize
    image = image.astype(np.float32, copy=False)  # shape of im = (224,224,3)

    return image