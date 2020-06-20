import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img,img_to_array
from keras.applications.vgg19 import preprocess_input
import numpy as np
import gensim
import cv2
from keras.models import load_model
import tensorflow as tf
import pickle

# load word2vec
word2vec_model = gensim.models.Word2Vec.load('ar_wiki_word2vec') # load word embedding model
# load label encoder
pkl_file = open('Label_encoder_yes_no.pkl', 'rb')
encoder = pickle.load(pkl_file)
pkl_file.close()
# pre-processing image
def image_preprocessing(image_path):
    '''
    Description: processing image (resize, normalize)

    Arguments:
        image_path: string - the image path
    Returns:
       image: a vector after done pre-processing
    '''

    image= cv2.imread(image_path)
    image= cv2.resize(image, (224, 224))
    image = image.astype(np.float32)/255  # Normlaize
    image= image.astype(np.float32, copy=False) # shape of im = (224,224,3)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image



image = image_preprocessing('/Users/afnanbq/PycharmProjects/AVQA/cow.jpg')
image = np.reshape(image,(1,image.shape[0],image.shape[1],image.shape[2]))
image.shape

# pre-processing question
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
    vec= np.zeros(dim) # initialization a list with zeros
    if token not in n_model.wv:
        _count= 0
        is_vec= True
        for w in token.split(" "):
            # search for the token in word2vec model
            if w in n_model.wv:
                _count += 1
                vec += n_model.wv[w]
        if _count > 0:
            vec= vec / _count
    else:
        vec= n_model.wv[token]
    return vec

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

    token_vector= np.reshape(token_vector, (1, len(token_vector)*300)) # reshape array to 1 and token vector lenght
    padded_inputs= tf.keras.preprocessing.sequence.pad_sequences(token_vector, dtype='float32',
                                                              padding='post', maxlen=3000)  # paadding vector
    padded_inputs= np.array(padded_inputs)

    return padded_inputs


tokens = "أهذه بقرة".split(' ')
print(tokens)

question = question_preprocessing(tokens)
question = question.reshape(1, question.shape[0], question.shape[1])
question.shape
# load model
model = load_model('/Users/afnanbq/PycharmProjects/AVQA/models/yes_no_resnet_withdropout.hdf5')
# predict
predict = model.predict([image,question])
print(predict)
classes = np.argmax(predict, axis=1)
print(classes)
print(encoder.inverse_transform(classes))
plt.imshow(image[0])
