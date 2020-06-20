from model import *
from prepare_data import *
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import os
import argparse


def get_model(dropout_rate):
    '''
    Description: import VQA model from model file

    Arguments:
        dropout_rate: droupout
    Returns:
        model: VQA model
    '''

    model = VQA(dropout_rate)
    return model


def train(args):
    '''
    Description: train VQA model

    Arguments:
        args: all arguments from the main class
    Returns:
        None
    '''
    dropout_rate = 0.5
    train_X, train_y = load_data(args.train_file_name, args.type)
    model = get_model(dropout_rate)
    checkpointer = ModelCheckpoint(filepath="/Users/afnanbq/PycharmProjects/AVQA/models/yes_no_resnet_withdropout.hdf5",monitor='loss', save_best_only=True, verbose=1)

    model.fit([train_X[0], train_X[1]],train_y, nb_epoch=args.epoch, batch_size=args.batch_size,
              callbacks=[checkpointer])


def test(args):
    '''
    Description: train VQA model

    Arguments:
        args: all arguments from the main class
    Returns:
        None
    '''

    model = load_model(args.model_name)
    test_X, test_y = load_data(args.test_file_name, 'test')

    scores = model.evaluate([test_X[0], test_X[1]], test_y)
    print("Accuracy: %.2f%%" % (scores[1] * 100))


if __name__ == "__main__":
    parser = argparse.ArgumentParser( )
    parser.add_argument('--type', type=str, default='train')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--train_file_name', type=str, default='yes_no_train.json')
    parser.add_argument('--model_name', type=str, default='/Users/afnanbq/PycharmProjects/AVQA/models/yes_no_resnet_withdropout.hdf5')
    parser.add_argument('--test_file_name', type=str, default='yes_no_test.json')


    args = parser.parse_args()

    if args.type == 'train':
        train(args)
    elif args.type == 'test':
        test(args)
