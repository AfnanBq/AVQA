from keras.layers import Dense, Activation, Dropout, LSTM
from keras.models import Sequential, Model
#from keras.applications import vgg16, vgg19
from keras.applications.resnet import ResNet50, ResNet152
from keras import optimizers
from keras.layers import *


def Language_model(dropout_rate):
    '''
    Description: build Language model

    Arguments:
        dropout_rate: dropout
    Returns:
        Language model
    '''

    print("Create Language Model ...")

    model = Sequential( )
    model.add(LSTM(units=512, return_sequences=True, input_shape=(None, 3000)))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(512, return_sequences=False))
    model.add(Dropout(dropout_rate))

    model.add(Dense(1024))
    model.add(Activation('tanh'))

    return model


def Image_model():
    '''
    Description: build Image model

    Arguments:

    Returns:
        Pretrained image model
    '''
    print("Create Image Model ...")

    model = ResNet152()
    model.input

    model.layers.pop( )
    new_layer = Dense(1024, name='FC-1024')
    inp = model.input
    out = new_layer(model.layers[-1].output)

    model = Model(inp, out)

    return model


def VQA(dropout_rate):
    '''
    Description: build VQA model by marging Language and Image model.

    Arguments:
        dropout_rate: dropout
    Returns:
        VQA model
    '''

    language_model = Language_model(dropout_rate)
    image_model = Image_model( )

    print("Create VQA Model ...")

    merge_model = Multiply( )([image_model.output, language_model.output])
    for i in range(2):
        merge_model = (Dense(1000, ))(merge_model)
        merge_model = (Activation('tanh'))(merge_model)
        merge_model = (Dropout(dropout_rate))(merge_model)


    merge_model = (Dense(2, ))(merge_model)

    merge_model = (Activation('softmax'))(merge_model)

    model = Model([image_model.input, language_model.input], merge_model)
    adam = optimizers.Adam()
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model

